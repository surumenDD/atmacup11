import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, List

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm.auto import tqdm

import wandb
from utils.env import EnvConfig
from utils.logger import get_logger
from utils.timing import trace

LOGGER = None
WANDB_PROJECT_NAME = "atmacup11"


# ==============================
# Config 定義
# ==============================

@dataclass
class ExpConfig:
    debug: bool = False
    seed: int = 42
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 800
    img_size: int = 224
    num_workers: int = 4
    weight_decay: float = 0.0
    # 学習に使う fold
    folds: List[int] = field(default_factory=lambda: [0])
    # mean/std 計算時にサンプリングする枚数（None なら全件）
    mean_std_sample_limit: Optional[int] = None


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    exp: ExpConfig = field(default_factory=ExpConfig)


cs = ConfigStore.instance()
cs.store(name="default", group="env", node=EnvConfig)
cs.store(name="default", group="exp", node=ExpConfig)


# ==============================
# Utility 関数
# ==============================

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_input_dir(cfg_input_dir: str) -> Path:
    """Kami さん実装と同じような入力ディレクトリ解決ロジック。"""
    cfg_dir = Path(cfg_input_dir)
    if cfg_dir.exists():
        return cfg_dir

    workspace_input = Path(__file__).resolve().parents[2] / "input"
    if workspace_input.exists():
        return workspace_input

    cwd_input = Path.cwd() / "input"
    return cwd_input


def compute_channel_mean_std(
    image_paths: List[Path],
    img_size: int,
    sample_limit: Optional[int] = None,
) -> tuple[List[float], List[float]]:
    """
    ノートブックの compute_channel_mean_std を元にした channel-wise mean/std 計算。
    """
    tfm = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),  # [0,1] スケール
    ])

    n = 0
    csum = torch.zeros(3, dtype=torch.float64)
    csum_sq = torch.zeros(3, dtype=torch.float64)

    if sample_limit is not None:
        paths = image_paths[:sample_limit]
    else:
        paths = image_paths

    for p in tqdm(paths, desc="Compute mean/std"):
        try:
            x = tfm(Image.open(p).convert("RGB"))
        except Exception:
            continue
        num_pix = x.shape[1] * x.shape[2]
        n += num_pix
        x_2d = x.reshape(3, -1)
        csum += x_2d.sum(dim=1).double()
        csum_sq += (x_2d ** 2).sum(dim=1).double()

    mean = (csum / n).tolist()
    var = (csum_sq / n - (csum / n) ** 2).tolist()
    std = [float(np.sqrt(max(v, 0.0))) for v in var]
    return mean, std


class AtmaDataset(Dataset):
    """
    ノートブック版 AtmaDataset を、パス生成と mean/std 正規化込みで実装。
    object_id からパスを作り、is_train で Augmentation 切り替え。
    """

    def __init__(
        self,
        meta_df: pd.DataFrame,
        photos_dir: Path,
        img_size: int,
        mean: Optional[List[float]],
        std: Optional[List[float]],
        is_train: bool = True,
    ) -> None:
        self.meta_df = meta_df.reset_index(drop=True)
        self.photos_dir = Path(photos_dir)
        self.img_size = img_size
        self.is_train = is_train
        self.mean = mean
        self.std = std

        tfms: List[torch.nn.Module] = [T.Resize((img_size, img_size))]
        if is_train:
            tfms.append(T.RandomHorizontalFlip(p=0.5))
        tfms.append(T.ToTensor())
        if self.mean is not None and self.std is not None:
            tfms.append(T.Normalize(mean=self.mean, std=self.std))

        self.transformer = T.Compose(tfms)

    def __len__(self) -> int:
        return len(self.meta_df)

    def __getitem__(self, idx: int):
        row = self.meta_df.iloc[idx]
        object_id = str(row["object_id"])
        img_path = self.photos_dir / f"{object_id}.jpg"
        img = Image.open(img_path).convert("RGB")
        img = self.transformer(img)

        if "target" in self.meta_df.columns:
            label = float(row["target"])
        else:
            label = -1.0

        return img, torch.tensor(label, dtype=torch.float32)


def create_resnet18d_reg(model_name: str = "resnet18d") -> nn.Module:
    """
    ノートブックの create_model と同等：
    - timm の resnet18d
    - pretrained=False
    - 出力 1 次元の回帰
    """
    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=1,  # そのまま 1 出力ヘッド
        in_chans=3,
    )
    return model


class RegressionModule(pl.LightningModule):
    """
    resnet18d による回帰タスク（RMSE）。
    """

    def __init__(
        self,
        model_name: str,
        learning_rate: float,
        weight_decay: float,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = create_resnet18d_reg(model_name=model_name)
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(1)  # (N, 1) -> (N,)

    def training_step(self, batch, batch_idx: int):
        imgs, targets = batch
        preds = self(imgs)
        loss = self.loss_fn(preds, targets)
        self.log("train_loss", loss, prog_bar=False,
                 on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        imgs, targets = batch
        preds = self(imgs)
        loss = self.loss_fn(preds, targets)
        rmse = torch.sqrt(self.loss_fn(preds, targets))
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_rmse", rmse, prog_bar=True, on_step=False, on_epoch=True)
        return {"val_loss": loss, "val_rmse": rmse}

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        imgs, _ = batch
        preds = self(imgs)
        return preds

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer


# ==============================
# Fold 学習（OOF 付き）
# ==============================

def train_one_fold(
    fold: int,
    train_df: pd.DataFrame,
    photos_dir: Path,
    cfg: Config,
    output_dir: Path,
    accelerator: str,
    devices: Optional[int],
    img_mean: List[float],
    img_std: List[float],
) -> tuple[np.ndarray, float, str]:
    """
    1 つの fold を学習して:
      - この fold の valid 行に対応する OOF 予測 (np.ndarray)
      - fold RMSE
      - ベスト checkpoint パス
    を返す（Kami さんの train_one_fold スタイル）。
    """
    df_trn = train_df[train_df["fold"] != fold].copy()
    df_val = train_df[train_df["fold"] == fold].copy()

    LOGGER.info("Train size (fold != %d): %d", fold, len(df_trn))
    LOGGER.info("Valid size (fold == %d): %d", fold, len(df_val))

    trn_ds = AtmaDataset(
        meta_df=df_trn,
        photos_dir=photos_dir,
        img_size=cfg.exp.img_size,
        mean=img_mean,
        std=img_std,
        is_train=True,
    )
    val_ds = AtmaDataset(
        meta_df=df_val,
        photos_dir=photos_dir,
        img_size=cfg.exp.img_size,
        mean=img_mean,
        std=img_std,
        is_train=False,
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
        model_name=cfg.exp.model_name if hasattr(
            cfg.exp, "model_name") else "resnet18d",
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
        project=WANDB_PROJECT_NAME,
        name=f"exp002_resnet18d_regnorm_fold{fold}",
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

    preds_batches = trainer.predict(
        module, dataloaders=val_loader, ckpt_path=best_path)
    preds = torch.cat(preds_batches, dim=0).cpu().numpy().reshape(-1)

    y_true = df_val["target"].values.astype(np.float32)
    rmse = float(np.sqrt(np.mean((preds - y_true) ** 2)))
    LOGGER.info("Fold %d RMSE (from best checkpoint): %.4f", fold, rmse)

    return preds, rmse, best_path


# ==============================
# Test 推論
# ==============================

def predict_test(
    fold_to_ckpt: Dict[int, Path],
    photos_dir: Path,
    test_df: pd.DataFrame,
    cfg: Config,
    accelerator: str,
    devices: Optional[int],
    img_mean: List[float],
    img_std: List[float],
) -> np.ndarray:
    """
    学習済み checkpoint 群から test を推論し、fold 平均した予測ベクトルを返す。
    """
    test_ds = AtmaDataset(
        meta_df=test_df.reset_index(drop=True),
        photos_dir=photos_dir,
        img_size=cfg.exp.img_size,
        mean=img_mean,
        std=img_std,
        is_train=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.exp.batch_size,
        shuffle=False,
        num_workers=cfg.exp.num_workers,
        pin_memory=True,
    )

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=False,
        enable_checkpointing=False,
    )

    fold_preds: List[np.ndarray] = []

    for fold, ckpt_path in fold_to_ckpt.items():
        LOGGER.info("Load checkpoint for fold %d: %s", fold, ckpt_path)
        module = RegressionModule.load_from_checkpoint(
            checkpoint_path=str(ckpt_path),
            model_name=cfg.exp.model_name if hasattr(
                cfg.exp, "model_name") else "resnet18d",
            learning_rate=cfg.exp.learning_rate,
            weight_decay=cfg.exp.weight_decay,
        )
        preds_batches = trainer.predict(module, dataloaders=test_loader)
        preds = torch.cat(preds_batches, dim=0).cpu().numpy().reshape(-1)
        LOGGER.info("Fold %d test preds shape: %s", fold, preds.shape)
        fold_preds.append(preds)

    if not fold_preds:
        raise RuntimeError("No checkpoints found. Cannot predict test.")

    if len(fold_preds) == 1:
        return fold_preds[0]
    else:
        return np.mean(np.stack(fold_preds, axis=0), axis=0)


# ==============================
# main
# ==============================

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: Config) -> None:
    global LOGGER

    exp_name = f"{Path(sys.argv[0]).parent.name}/{HydraConfig.get().runtime.choices.exp}"
    base_output_dir = Path(cfg.env.exp_output_dir)
    output_dir = base_output_dir / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER = get_logger(__name__, output_dir)
    LOGGER.info("Start experiment: %s", exp_name)
    LOGGER.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    set_seed(cfg.exp.seed)
    if torch.cuda.is_available():
        # Tensor Core を使うための推奨設定
        torch.set_float32_matmul_precision("high")

    input_dir = resolve_input_dir(cfg.env.input_dir)
    photos_dir = input_dir / "photos"
    train_csv = input_dir / "train.csv"
    test_csv = input_dir / "test.csv"
    sample_sub_path = input_dir / "atmaCup#11_sample_submission.csv"

    LOGGER.info("Input dir: %s", input_dir)
    LOGGER.info("Train CSV: %s", train_csv)
    LOGGER.info("Test  CSV: %s", test_csv)
    LOGGER.info("Photos dir: %s", photos_dir)

    with trace("load_csv"):
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)

    LOGGER.info("train_df shape: %s", train_df.shape)
    LOGGER.info("test_df  shape: %s", test_df.shape)
    LOGGER.info("train_df columns: %s", list(train_df.columns))

    # 型をそろえる
    train_df["object_id"] = train_df["object_id"].astype(str)
    test_df["object_id"] = test_df["object_id"].astype(str)
    train_df["art_series_id"] = train_df["art_series_id"].astype(str)
    train_df["target"] = train_df["target"].astype(float)

    # ==========
    # mean/std 計算
    # ==========
    image_paths = [photos_dir /
                   f"{oid}.jpg" for oid in train_df["object_id"].tolist()]

    with trace("compute_mean_std"):
        img_mean, img_std = compute_channel_mean_std(
            image_paths=image_paths,
            img_size=cfg.exp.img_size,
            sample_limit=cfg.exp.mean_std_sample_limit,
        )

    LOGGER.info("Computed mean: %s", img_mean)
    LOGGER.info("Computed std : %s", img_std)

    # ==========
    # StratifiedGroupKFold で fold 割り当て
    # ==========
    LOGGER.info("Create folds with StratifiedGroupKFold")

    n_splits = 5
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=cfg.exp.seed,
    )

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

    # ==========
    # fold 学習 & OOF 作成
    # ==========
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info("Using device: %s", device_str)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices: Optional[int] = 1 if torch.cuda.is_available() else None

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
                img_mean=img_mean,
                img_std=img_std,
            )

        mask = train_df["fold"].values == fold
        oof_pred[mask] = oof_pred_fold

        fold_to_ckpt[fold] = Path(best_ckpt)
        fold_scores[fold] = fold_rmse
        LOGGER.info("Fold %d RMSE: %.4f", fold, fold_rmse)

    LOGGER.info("All folds training finished. Scores: %s", fold_scores)

    # OOF スコア
    used_folds = cfg.exp.folds
    mask_used = train_df["fold"].isin(used_folds).values
    y_true_all = train_df["target"].values.astype(np.float32)
    oof_rmse = float(
        np.sqrt(np.mean((oof_pred[mask_used] - y_true_all[mask_used]) ** 2)))
    LOGGER.info("OOF RMSE (using folds=%s): %.4f", used_folds, oof_rmse)

    oof_df = pd.DataFrame(
        {
            "object_id": train_df["object_id"],
            "fold": train_df["fold"],
            "target": train_df["target"],
            "oof_pred": oof_pred,
        }
    )
    oof_path = output_dir / f"oof_resnet18d_exp002_{len(used_folds)}folds.csv"
    oof_df.to_csv(oof_path, index=False)
    LOGGER.info("Saved OOF to %s", oof_path)

    # ==========
    # Test 推論 + submission
    # ==========
    if not sample_sub_path.exists():
        LOGGER.error("Sample submission not found: %s", sample_sub_path)
        return

    submission_df = pd.read_csv(sample_sub_path)
    LOGGER.info("Loaded sample submission: %s (shape=%s)",
                sample_sub_path, submission_df.shape)

    with trace("predict_test"):
        test_pred = predict_test(
            fold_to_ckpt=fold_to_ckpt,
            photos_dir=photos_dir,
            test_df=test_df,
            cfg=cfg,
            accelerator=accelerator,
            devices=devices,
            img_mean=img_mean,
            img_std=img_std,
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
        f"submission_resnet18d_exp002_{len(fold_to_ckpt)}folds.csv"
    submission_df.to_csv(sub_path, index=False)
    LOGGER.info("Saved submission to %s", sub_path)
    LOGGER.info("Done.")


if __name__ == "__main__":
    main()
