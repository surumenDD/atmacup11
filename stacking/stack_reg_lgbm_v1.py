# stacking/stack_reg_lgbm_v1.py

import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_squared_error

# =========================
# 設定
# =========================

# 実験名とファイルパスのテンプレート
EXPERIMENTS = [
    "exp004_resnet18d_aug",
    "exp005_dino_vits16_reg",
    "exp006_dino_vitb16_reg",
    "exp007_dino_resnet18_reg",
]

BASE_OOF_TEMPLATE = (
    "output/experiments/{exp}/default/"
    "oof_{exp}_5folds.csv"
)
BASE_SUB_TEMPLATE = (
    "output/experiments/{exp}/default/"
    "submission_{exp}_5folds.csv"
)

# 出力先
STACKING_OUTPUT_DIR = Path("output/stacking")
STACKING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STACKING_OOF_PATH = STACKING_OUTPUT_DIR / "oof_stack_reg_lgbm_v1_5folds.csv"
STACKING_SUB_PATH = STACKING_OUTPUT_DIR / \
    "submission_stack_reg_lgbm_v1_5folds.csv"

RANDOM_STATE = 42

# =========================
# ロード系ユーティリティ
# =========================


def load_oof_and_sub(exp_name: str) -> Dict[str, pd.DataFrame]:
    """
    1つの実験について OOF と submission を読み込む。
    """
    oof_path = BASE_OOF_TEMPLATE.format(exp=exp_name)
    sub_path = BASE_SUB_TEMPLATE.format(exp=exp_name)

    if not Path(oof_path).exists():
        raise FileNotFoundError(f"OOF not found: {oof_path}")
    if not Path(sub_path).exists():
        raise FileNotFoundError(f"SUB not found: {sub_path}")

    oof_df = pd.read_csv(oof_path)
    sub_df = pd.read_csv(sub_path)

    print(
        f"[INFO] loaded {exp_name}:\n"
        f"       OOF: {oof_path}\n"
        f"       SUB: {sub_path}"
    )

    return {
        "exp_name": exp_name,
        "oof": oof_df,
        "sub": sub_df,
    }


def build_meta_features(
    exp_results: List[Dict[str, pd.DataFrame]]
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """
    meta 学習用の特徴量とターゲット、fold 情報を作成。

    戻り値:
        X_oof : (N_train, n_models) の特徴量
        y     : (N_train,) の target
        folds : (N_train,) の fold 番号
    """
    # ベースとして最初の OOF を使用
    base_oof = exp_results[0]["oof"].copy()
    # 必須カラムの存在確認
    for col in ["object_id", "fold", "target", "oof_pred"]:
        if col not in base_oof.columns:
            raise KeyError(f"[ERROR] {col} not found in base OOF.")

    # object_id, fold, target を保持
    meta_df = base_oof[["object_id", "fold", "target"]].copy()
    meta_df = meta_df.reset_index(drop=True)

    # それぞれの実験の oof_pred を特徴量として追加
    for res in exp_results:
        exp_name = res["exp_name"]
        oof_df = res["oof"].copy()

        # object_id / fold / target の整合性を確認する
        # （順序が異なる可能性があるため、sort してから比較）
        cols_key = ["object_id", "fold", "target"]
        base_key = base_oof[cols_key].sort_values(
            cols_key).reset_index(drop=True)
        curr_key = oof_df[cols_key].sort_values(
            cols_key).reset_index(drop=True)

        if not base_key.equals(curr_key):
            raise ValueError(
                f"[ERROR] OOF rows mismatch between base and {exp_name}."
            )

        # 元の順序で meta_df に join するため、index ベースでそのまま使う
        feature_col = f"oof_{exp_name}"
        meta_df[feature_col] = oof_df["oof_pred"].values

    # 特徴量行列とターゲットと fold を取り出す
    feature_cols = [c for c in meta_df.columns if c.startswith("oof_")]
    X_oof = meta_df[feature_cols].copy()
    y = meta_df["target"].astype(float).copy()
    folds = meta_df["fold"].values.copy()

    print(f"[INFO] folds: {sorted(set(folds))}")
    print(f"[INFO] base models: {feature_cols}")

    return X_oof, y, folds


def build_meta_test_features(exp_results: List[Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """
    テストデータに対する meta 特徴量を作成。
    各 submission.csv の target 列を特徴量とする。
    """
    # ベースは最初の submission とし、長さだけ確認
    base_sub = exp_results[0]["sub"].copy()
    n_test = len(base_sub)

    meta_test = pd.DataFrame(index=np.arange(n_test))

    for res in exp_results:
        exp_name = res["exp_name"]
        sub_df = res["sub"].copy()

        if len(sub_df) != n_test:
            raise ValueError(
                f"[ERROR] test length mismatch: base={n_test}, {exp_name}={len(sub_df)}"
            )

        if "target" not in sub_df.columns:
            raise KeyError(
                f"[ERROR] column 'target' not found in submission of {exp_name}")

        feature_col = f"sub_{exp_name}"
        meta_test[feature_col] = sub_df["target"].values

    print(f"[INFO] test meta features shape: {meta_test.shape}")
    return meta_test


# =========================
# LGBM で stacking
# =========================

def run_stacking():
    # 1. 各実験の OOF と SUB を読み込み
    exp_results: List[Dict[str, pd.DataFrame]] = []
    for exp in EXPERIMENTS:
        res = load_oof_and_sub(exp)
        exp_results.append(res)

    # 2. meta 用 OOF 特徴量と y, fold を作る
    X_oof, y, folds = build_meta_features(exp_results)
    meta_test = build_meta_test_features(exp_results)

    # 3. LGBMRegressor のハイパーパラメータ
    lgbm_params = {
        "n_estimators": 10000,
        "learning_rate": 0.01,
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 31,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "subsample_freq": 1,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    unique_folds = sorted(set(folds))
    n_samples = len(X_oof)
    n_test = len(meta_test)

    oof_meta = np.zeros(n_samples, dtype=np.float32)
    test_meta_preds = np.zeros((len(unique_folds), n_test), dtype=np.float32)

    # 4. fold ごとに LGBM を学習
    for i, fold in enumerate(unique_folds):
        print(f"[INFO] ==== meta fold {fold} training start ====")

        train_idx = folds != fold
        valid_idx = folds == fold

        X_tr = X_oof.iloc[train_idx].values
        y_tr = y.iloc[train_idx].values
        X_val = X_oof.iloc[valid_idx].values
        y_val = y.iloc[valid_idx].values

        meta = LGBMRegressor(**lgbm_params)

        # callbacks で early_stopping とログを指定（fit の verbose 引数や early_stopping_rounds は使わない）
        callbacks = [
            early_stopping(stopping_rounds=100, verbose=False),
            log_evaluation(period=100),
        ]

        meta.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=callbacks,
        )

        # validation OOF
        val_pred = meta.predict(X_val, num_iteration=meta.best_iteration_)
        oof_meta[valid_idx] = val_pred.astype(np.float32)

        # test 予測（fold ごとに持っておいて最後に平均）
        test_pred = meta.predict(
            meta_test.values, num_iteration=meta.best_iteration_)
        test_meta_preds[i, :] = test_pred.astype(np.float32)

        rmse_fold = np.sqrt(mean_squared_error(y_val, val_pred))
        print(f"[INFO] fold {fold} RMSE (meta): {rmse_fold:.4f}")

    # 5. OOF 全体のスコア
    oof_rmse = np.sqrt(mean_squared_error(y, oof_meta))
    print(f"[INFO] OOF RMSE (meta, all folds): {oof_rmse:.6f}")

    # 6. OOF 結果を保存
    #   object_id, fold, target は exp_results[0]["oof"] から拝借
    base_oof = exp_results[0]["oof"].copy()
    out_oof_df = pd.DataFrame(
        {
            "object_id": base_oof["object_id"],
            "fold": base_oof["fold"],
            "target": base_oof["target"],
            "oof_pred": oof_meta,
        }
    )
    out_oof_df.to_csv(STACKING_OOF_PATH, index=False)
    print(f"[INFO] saved meta OOF to: {STACKING_OOF_PATH}")

    # 7. test 予測を fold 平均して submission を作成
    test_pred_mean = test_meta_preds.mean(axis=0)

    # sample_submission の雛形を使って出力（1列 target のみ）
    sub_df = pd.DataFrame({"target": test_pred_mean})
    sub_df.to_csv(STACKING_SUB_PATH, index=False)
    print(f"[INFO] saved meta SUB to: {STACKING_SUB_PATH}")


if __name__ == "__main__":
    run_stacking()
