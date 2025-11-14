# experiments/exp001_resnet18d_reg/run.py
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import hydra
import numpy as np
import pandas as pd
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

import timm
import wandb

from typing import Dict, Optional

from utils.env import EnvConfig
from utils.logger import get_logger
from utils.timing import trace

# ------------------
# 設定クラス
# ------------------


@dataclass
class ExpConfig:
    debug: bool = False
    seed: int = 42
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 5
    img_size: int = 224
    num_workers: int = 4
    weight_decay: float = 0.0
    model_name: str = "resnet18d"
    folds: List[int] = field(default_factory=lambda: [0])


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    exp: ExpConfig = field(default_factory=ExpConfig)


cs = ConfigStore.instance()
cs.store(name="default", group="env", node=EnvConfig)
cs.store(name="default", group="exp", node=ExpConfig)

LOGGER = None


def set_seed(seed: int) -> None:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_input_dir(cfg_input_dir: str) -> Path:
    """
    Config の input_dir を優先しつつ、
    なければプロジェクト直下の input/ を探す。
    """
    cfg_dir = Path(cfg_input_dir)
    if cfg_dir.exists():
        return cfg_dir
    # run.py から見て 2 つ上がプロジェクトルートの想定
    workspace_input = Path(__file__).resolve().parents[2] / "input"
    if workspace_input.exists():
        return workspace_input
    # 最後の手段: CWD/input
    return Path.cwd() / "input"


def create_transforms(img_size: int) -> tuple[T.Compose, T.Compose]:
    """
    学習用・検証用のデータ変換を返す。
    まずはシンプルに Resize + Flip + ToTensor のみ。
    （Normalize や高度な拡張は後で追加する）
    """
    train_tfm = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
        ]
    )
    valid_tfm = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ]
    )
    return train_tfm, valid_tfm


class AtmaDataset(Dataset):
    """
    train/test 共通で使う Dataset。
    - meta_df: DataFrame。train のときは target 列を含む。
    - images_root: photos ディレクトリの Path。
    - is_train: True のときは target を返す。
    """

    def __init__(
        self,
        meta_df: pd.DataFrame,
        images_root: Path,
        transform: T.Compose,
        is_train: bool = True,
    ) -> None:
        self.meta_df = meta_df.reset_index(drop=True)
        self.images_root = Path(images_root)
        self.transform = transform
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.meta_df)

    def __getitem__(self, idx: int):
        row = self.meta_df.iloc[idx]
        object_id = str(row["object_id"])
        img_path = self.images_root / f"{object_id}.jpg"

        # 念のため他拡張子も見る
        if not img_path.exists():
            for ext in [".jpeg", ".png"]:
                alt = self.images_root / f"{object_id}{ext}"
                if alt.exists():
                    img_path = alt
                    break

        with Image.open(img_path) as img:
            img = img.convert("RGB")

        img = self.transform(img)

        if self.is_train:
            target = float(row["target"])
            return img, torch.tensor(target, dtype=torch.float32)
        else:
            return img, object_id


class RegressionModule(pl.LightningModule):
    """
    timm の resnet18d を使った回帰モデル（target をそのまま予測）。
    MSELoss を最小化しつつ、RMSE をログに出す。
    """

    def __init__(self, model_name: str, learning_rate: float, weight_decay: float) -> None:
        super().__init__()
        self.save_hyperparameters()  # hparams に保存（W&B からも見える）

        # timm モデルをベースに最終層だけ 1 出力に付け替える
        backbone = timm.create_model(
            model_name, pretrained=False, num_classes=0, in_chans=3)
        in_features = backbone.num_features  # resnet18d は 512
        backbone.fc = nn.Linear(in_features, 1)  # 1 次元の回帰

        self.model = backbone
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).view(-1)  # (N,) に揃える

    def training_step(self, batch, batch_idx: int):
        imgs, targets = batch  # targets: (N,)
        preds = self(imgs)     # (N,)
        loss = self.loss_fn(preds, targets)
        self.log("train_loss", loss, prog_bar=False,
                 on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        imgs, targets = batch
        preds = self(imgs)
        loss = self.loss_fn(preds, targets)
        rmse = torch.sqrt(torch.mean((preds - targets) ** 2))
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_rmse", rmse, prog_bar=True, on_step=False, on_epoch=True)
        return {"val_loss": loss, "val_rmse": rmse}

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        imgs, _ = batch
        preds = self(imgs)
        return preds  # (N,)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer


def train_one_fold(
    fold: int,
    train_df: pd.DataFrame,
    photos_dir: Path,
    cfg: Config,
    output_dir: Path,
    accelerator: str,
    devices: Optional[int],
    train_tfm: T.Compose,
    valid_tfm: T.Compose,
) -> tuple[np.ndarray, float, str]:
    """
    この fold の学習を行い、
    - df_val に対応する予測ベクトル（OOF用）
    - fold の RMSE
    - ベストcheckpointのパス
    を返す。
    """
    # === fold の分割 ===
    df_trn = train_df[train_df["fold"] != fold].copy()
    df_val = train_df[train_df["fold"] == fold].copy()

    LOGGER.info("Train size (fold != %d): %d", fold, len(df_trn))
    LOGGER.info("Valid size (fold == %d): %d", fold, len(df_val))

    trn_ds = AtmaDataset(
        meta_df=df_trn,
        images_root=photos_dir,
        transform=train_tfm,
        is_train=True,
    )
    val_ds = AtmaDataset(
        meta_df=df_val,
        images_root=photos_dir,
        transform=valid_tfm,
        is_train=True,
    )

    trn_loader = DataLoader(
        trn_ds,
        batch_size=cfg.exp.batch_size,
        shuffle=True,
        num_workers=cfg.exp.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.exp.batch_size,
        shuffle=False,
        num_workers=cfg.exp.num_workers,
        pin_memory=True,
    )

    module = RegressionModule(
        model_name=cfg.exp.model_name,
        learning_rate=cfg.exp.learning_rate,
        weight_decay=cfg.exp.weight_decay,
    )

    fold_out_dir = output_dir / f"fold{fold}"
    fold_out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        dirpath=str(fold_out_dir),
        filename="model-best",
        monitor="val_rmse",
        mode="min",
        save_top_k=1,
        save_weights_only=True,
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")

    if cfg.exp.debug:
        os.environ["WANDB_MODE"] = "disabled"

    wandb_logger = WandbLogger(
        project="atmacup11",
        name=f"exp001_resnet18d_reg_fold{fold}",
        save_dir=str(output_dir),
    )

    trainer = pl.Trainer(
        max_epochs=cfg.exp.epochs,
        accelerator=accelerator,
        devices=devices,
        logger=wandb_logger,
        callbacks=[ckpt_cb, lr_cb],
        log_every_n_steps=10,
    )

    trainer.fit(module, train_dataloaders=trn_loader,
                val_dataloaders=val_loader)

    best_path = ckpt_cb.best_model_path
    LOGGER.info("Fold %d best checkpoint: %s", fold, best_path)

    # === ベストcheckpointで valid を推論 ===
    preds_batches = trainer.predict(
        module, dataloaders=val_loader, ckpt_path=best_path)
    preds = torch.cat(preds_batches, dim=0).cpu(
    ).numpy().reshape(-1)   # ← このfoldの OOF 予測

    y_true = df_val["target"].values.astype(np.float32)
    rmse = float(np.sqrt(np.mean((preds - y_true) ** 2)))
    LOGGER.info("Fold %d RMSE (from best checkpoint): %.4f", fold, rmse)

    # preds（=この fold の valid 行に対応する OOF）も返す
    return preds, rmse, best_path


def predict_test(
    fold_to_ckpt: Dict[int, Path],
    photos_dir: Path,
    test_df: pd.DataFrame,
    cfg: Config,
    accelerator: str,
    devices: Optional[int],
    valid_tfm: T.Compose,
) -> np.ndarray:
    """
    学習済みの fold ごとの checkpoint から test を推論し、
    fold 平均した予測ベクトルを返す（kamiさんの predict_test に相当）。
    """
    # === test Dataset / DataLoader ===
    test_ds = AtmaDataset(
        meta_df=test_df.reset_index(drop=True),
        images_root=photos_dir,
        transform=valid_tfm,
        is_train=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.exp.batch_size,
        shuffle=False,
        num_workers=cfg.exp.num_workers,
        pin_memory=True,
    )

    # 推論専用 Trainer（logger / checkpoint なし）
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=False,
        enable_checkpointing=False,
    )

    fold_preds: list[np.ndarray] = []

    for fold, ckpt_path in fold_to_ckpt.items():
        LOGGER.info("Load checkpoint for fold %d: %s", fold, ckpt_path)

        module = RegressionModule.load_from_checkpoint(
            checkpoint_path=str(ckpt_path),
            model_name=cfg.exp.model_name,
            learning_rate=cfg.exp.learning_rate,
            weight_decay=cfg.exp.weight_decay,
        )

        preds_batches = trainer.predict(module, dataloaders=test_loader)
        preds = torch.cat(preds_batches, dim=0).cpu().numpy().reshape(-1)
        LOGGER.info("Fold %d test preds shape: %s", fold, preds.shape)
        fold_preds.append(preds)

    if not fold_preds:
        raise RuntimeError(
            "No checkpoints found in fold_to_ckpt. Cannot predict test.")

    if len(fold_preds) == 1:
        return fold_preds[0]
    else:
        return np.mean(np.stack(fold_preds, axis=0), axis=0)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: Config) -> None:
    global LOGGER

    # 実験名を Hydra の情報から決める
    exp_name = f"{Path(__file__).parent.name}/{HydraConfig.get().runtime.choices.exp}"
    output_dir = Path(cfg.env.exp_output_dir) / exp_name
    os.makedirs(output_dir, exist_ok=True)

    # ロガー設定
    LOGGER = get_logger(__name__, output_dir)
    LOGGER.info("Start experiment: %s", exp_name)
    LOGGER.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # 乱数シード
    set_seed(cfg.exp.seed)

    # データ読み込み
    input_dir = resolve_input_dir(cfg.env.input_dir)
    train_csv = input_dir / "train.csv"
    test_csv = input_dir / "test.csv"

    LOGGER.info("Input dir: %s", input_dir)
    LOGGER.info("Train CSV: %s", train_csv)
    LOGGER.info("Test  CSV: %s", test_csv)

    with trace("load_csv"):
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)

    LOGGER.info("train_df shape: %s", train_df.shape)
    LOGGER.info("test_df  shape: %s", test_df.shape)
    LOGGER.info("train_df columns: %s", list(train_df.columns))

    # ==============================
    # Fold 作成 (StratifiedGroupKFold)
    # ==============================
    LOGGER.info("Create folds with StratifiedGroupKFold")

    n_splits = 5
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=cfg.exp.seed)

    # y: target, groups: art_series_id
    y = train_df["target"].values
    groups = train_df["art_series_id"].values
    fold_indices = np.zeros(len(train_df), dtype=int)

    for fold, (_, val_idx) in enumerate(sgkf.split(train_df, y, groups)):
        fold_indices[val_idx] = fold

    train_df["fold"] = fold_indices

    LOGGER.info("Fold distribution (by target):")
    fold_target_counts = train_df.groupby(
        "fold")["target"].value_counts().unstack().fillna(0)
    LOGGER.info("\n%s", fold_target_counts)

    # ==============================
    # Lightning で学習 (cfg.exp.folds に書かれた fold を順に学習)
    # ==============================
    photos_dir = input_dir / "photos"
    LOGGER.info("Photos dir: %s", photos_dir)

    train_tfm, valid_tfm = create_transforms(cfg.exp.img_size)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info("Using device: %s", device_str)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices: Optional[int] = 1 if torch.cuda.is_available() else None

    # === fold ごとの学習 ===
    oof_pred = np.zeros(len(train_df), dtype=np.float32)

    fold_to_ckpt: Dict[int, Path] = {}
    fold_scores: Dict[int, float] = {}

    for fold in cfg.exp.folds:
        LOGGER.info("========== Fold %d training start ==========", fold)
        with trace(f"train_fold_{fold}"):
            oof_pred_fold, fold_rmse, best_ckpt = train_one_fold(
                fold=fold,
                train_df=train_df,
                photos_dir=photos_dir,
                cfg=cfg,
                output_dir=output_dir,
                accelerator=accelerator,
                devices=devices,
                train_tfm=train_tfm,
                valid_tfm=valid_tfm,
            )

        # この fold の valid 行に OOF を書き込む
        mask = train_df["fold"].values == fold
        oof_pred[mask] = oof_pred_fold

        fold_to_ckpt[fold] = Path(best_ckpt)
        fold_scores[fold] = fold_rmse
        LOGGER.info("Fold %d RMSE: %.4f", fold, fold_rmse)

    LOGGER.info("All folds training finished. Scores: %s", fold_scores)

    # === OOF スコア計算 ===
    used_folds = cfg.exp.folds
    mask_used = train_df["fold"].isin(used_folds).values  # 一応、使ったfoldだけ
    y_true_all = train_df["target"].values.astype(np.float32)

    oof_rmse = float(
        np.sqrt(np.mean((oof_pred[mask_used] - y_true_all[mask_used]) ** 2))
    )
    LOGGER.info(
        "OOF RMSE (using folds=%s): %.4f",
        used_folds,
        oof_rmse,
    )

    # === OOF を CSV 保存 ===
    oof_df = pd.DataFrame(
        {
            "object_id": train_df["object_id"],
            "fold": train_df["fold"],
            "target": train_df["target"],
            "oof_pred": oof_pred,
        }
    )
    oof_path = output_dir / f"oof_resnet18d_exp001_{len(used_folds)}folds.csv"
    oof_df.to_csv(oof_path, index=False)
    LOGGER.info("Saved OOF to %s", oof_path)

    # ==============================
    # Test 予測 ＋ submission 作成
    # ==============================
    sample_path = input_dir / "atmaCup#11_sample_submission.csv"
    if not sample_path.exists():
        LOGGER.error("Sample submission not found: %s", sample_path)
        return

    submission_df = pd.read_csv(sample_path)
    LOGGER.info("Loaded sample submission: %s (shape=%s)",
                sample_path, submission_df.shape)

    with trace("predict_test"):
        test_pred = predict_test(
            fold_to_ckpt=fold_to_ckpt,
            photos_dir=photos_dir,
            test_df=test_df,
            cfg=cfg,
            accelerator=accelerator,
            devices=devices,
            valid_tfm=valid_tfm,
        )

    if len(submission_df) != len(test_pred):
        LOGGER.warning(
            "Length mismatch: submission_df=%d, preds=%d",
            len(submission_df),
            len(test_pred),
        )
        n = min(len(submission_df), len(test_pred))
        submission_df.loc[: n - 1, "target"] = test_pred[:n]
    else:
        submission_df["target"] = test_pred

    sub_path = output_dir / \
        f"submission_resnet18d_exp001_{len(fold_to_ckpt)}folds.csv"
    submission_df.to_csv(sub_path, index=False)
    LOGGER.info("Saved submission to %s", sub_path)


if __name__ == "__main__":
    main()
