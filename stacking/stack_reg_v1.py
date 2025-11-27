# stacking/stack_reg_v1.py

import os
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# =========================
# 設定
# =========================

# 回帰モデルとしてスタックしたい実験名
# ※ディレクトリ名に合わせて書き換えてください
BASE_EXPERIMENTS: List[str] = [
    "exp004_resnet18d_aug",       # resnet18d 回帰
    "exp005_dino_vits16_reg",     # DINO ViT small 回帰
    "exp006_dino_vitb16_reg",     # DINO ViT base 回帰（仮）
    "exp007_dino_resnet18_reg",   # DINO ResNet18 回帰（仮）
]

# experiments 配下のベースディレクトリ
EXP_BASE_DIR = Path("./output/experiments")

# sample submission のパス
INPUT_DIR = Path("./input")
SAMPLE_SUB_PATH = INPUT_DIR / "atmaCup#11_sample_submission.csv"

# 出力先
STACK_OUTPUT_DIR = Path("./output/stacking/stack_reg_v1")
STACK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# ユーティリティ
# =========================

def find_single_csv(directory: Path, prefix: str) -> Path:
    """
    directory 内で prefix から始まる csv を 1 つ探す。
    1 つもない / 複数ある場合はエラーにする。
    """
    candidates = sorted(directory.glob(f"{prefix}*.csv"))
    if len(candidates) == 0:
        raise FileNotFoundError(f"{directory} に {prefix}*.csv が見つかりません")
    if len(candidates) > 1:
        raise RuntimeError(
            f"{directory} に {prefix}*.csv が複数あります: {candidates}")
    return candidates[0]


# =========================
# OOF と test 予測の読み込み
# =========================

def load_oof_and_test() -> tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    すべての base 実験について
    - OOF: object_id, fold, target, oof_pred_<exp_name> を集約
    - test: 各モデルの test 予測 (N_test, n_models)
    を返す。
    """
    oof_merged: pd.DataFrame | None = None
    test_preds_list: List[np.ndarray] = []
    model_cols: List[str] = []

    for exp_name in BASE_EXPERIMENTS:
        exp_dir = EXP_BASE_DIR / exp_name / "default"
        if not exp_dir.exists():
            raise FileNotFoundError(f"実験ディレクトリがありません: {exp_dir}")

        # OOF
        oof_path = find_single_csv(exp_dir, f"oof_{exp_name}_")
        oof_df = pd.read_csv(oof_path)
        if not {"object_id", "fold", "target", "oof_pred"}.issubset(oof_df.columns):
            raise ValueError(f"{oof_path} のカラムが想定と違います: {oof_df.columns}")

        oof_df = oof_df[["object_id", "fold", "target", "oof_pred"]].copy()
        oof_col_name = f"oof_{exp_name}"
        oof_df.rename(columns={"oof_pred": oof_col_name}, inplace=True)

        # マージ
        if oof_merged is None:
            oof_merged = oof_df
        else:
            # object_id, fold, target で整合している前提
            oof_merged = oof_merged.merge(
                oof_df, on=["object_id", "fold", "target"], how="inner"
            )

        # test 予測
        sub_path = find_single_csv(exp_dir, f"submission_{exp_name}_")
        sub_df = pd.read_csv(sub_path)
        if "target" not in sub_df.columns:
            raise ValueError(f"{sub_path} に target カラムがありません")
        test_preds_list.append(sub_df["target"].values.astype(np.float32))
        model_cols.append(oof_col_name)

        print(f"[INFO] loaded {exp_name}:")
        print(f"       OOF: {oof_path}")
        print(f"       SUB: {sub_path}")

    assert oof_merged is not None
    # (N_train, n_models) の特徴量
    X_oof = oof_merged[model_cols].values.astype(np.float32)
    y_oof = oof_merged["target"].values.astype(np.float32)
    folds = oof_merged["fold"].values.astype(int)

    # test 側
    test_preds_arr = np.stack(test_preds_list, axis=1)  # (N_test, n_models)
    return oof_merged, test_preds_arr, model_cols


# =========================
# 2nd レベルのスタッキング (Ridge)
# =========================

def run_stacking():
    # 1) base モデルの OOF / test 読み込み
    oof_df, test_feats, model_cols = load_oof_and_test()
    X = oof_df[model_cols].values.astype(np.float32)
    y = oof_df["target"].values.astype(np.float32)
    folds = oof_df["fold"].values.astype(int)

    unique_folds = sorted(np.unique(folds))
    print(f"[INFO] folds: {unique_folds}")
    print(f"[INFO] base models: {model_cols}")

    # 2) 2nd レベル OOF を fold ごとに作る
    oof_stack = np.zeros_like(y, dtype=np.float32)
    test_stack_folds: List[np.ndarray] = []

    for f in unique_folds:
        tr_idx = folds != f
        va_idx = folds == f

        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va = X[va_idx]

        # ここではシンプルに Ridge(alpha=1.0)
        meta = Ridge(alpha=1.0, random_state=42)
        meta.fit(X_tr, y_tr)

        oof_stack[va_idx] = meta.predict(X_va).astype(np.float32)
        test_stack_folds.append(meta.predict(test_feats).astype(np.float32))

        print(f"[INFO] fold {f}: train={tr_idx.sum()}, valid={va_idx.sum()}")

    # fold ごとの test 予測を平均
    test_stack = np.mean(np.stack(test_stack_folds, axis=0), axis=0)

    # 3) OOF RMSE を計算
    rmse = float(np.sqrt(np.mean((oof_stack - y) ** 2)))
    print(f"[RESULT] Stacking OOF RMSE: {rmse:.4f}")

    # 4) OOF / submission を保存
    # OOF: object_id, fold, target, stack_pred, 各モデルの oof も残しておく
    oof_out = oof_df.copy()
    oof_out["stack_pred"] = oof_stack
    oof_out_path = STACK_OUTPUT_DIR / "stack_oof_reg_v1.csv"
    oof_out.to_csv(oof_out_path, index=False)
    print(f"[INFO] saved stacked OOF: {oof_out_path}")

    # submission: sample_submission の行数に合わせて target を差し替え
    if not SAMPLE_SUB_PATH.exists():
        raise FileNotFoundError(f"sample submission がありません: {SAMPLE_SUB_PATH}")
    sub_base = pd.read_csv(SAMPLE_SUB_PATH)
    if len(sub_base) != len(test_stack):
        raise RuntimeError(
            f"test 行数が一致しません: sample={len(sub_base)}, stack={len(test_stack)}"
        )

    sub_base["target"] = test_stack
    sub_out_path = STACK_OUTPUT_DIR / "submission_stack_reg_v1.csv"
    sub_base.to_csv(sub_out_path, index=False)
    print(f"[INFO] saved stacked submission: {sub_out_path}")


if __name__ == "__main__":
    run_stacking()
