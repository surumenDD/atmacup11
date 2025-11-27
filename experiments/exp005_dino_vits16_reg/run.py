import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, List, Tuple

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
EXP_SHORT_NAME = Path(__file__).parent.name

# ==============================
# Config 定義
# ==============================


@dataclass
class ExpConfig:
    debug: bool = False
    seed: int = 42
    learning_rate: float = 1e-4
    batch_size: int = 64
    epochs: int = 100
    img_size: int = 224
    num_workers: int = 4
    weight_decay: float = 0.0
    folds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])

    # ==== モデル関連 ====
    model_family: str = "dino"          # "resnet" or "dino"
    # 例: "resnet18d", "vit_small", "vit_base", "resnet50_dino" など
    backbone_name: str = "vit_small"
    task: str = "reg"                   # "reg" or "cls"
    num_classes: int = 4                # cls のときだけ使用
    freeze_backbone: bool = False       # full fine-tune をデフォルト

    # ==== DINO 用設定 ====
    dino_repo_dir: str = "./external/dino"
    dino_checkpoint_path: str = "./output/dino/vit_small_ep800/checkpoint.pth"
    dino_checkpoint_key: str = "teacher"
    dino_patch_size: int = 16

    # embed_dim は backbone_name から決める
    dino_embed_dim_small: int = 384
    dino_embed_dim_base: int = 768

    # ==== 画像統計 ====
    mean_std_sample_limit: Optional[int] = None

    # ==== EMA ====
    use_ema: bool = True
    ema_decay: float = 0.999


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
    """入力ディレクトリの優先順位を決める。"""
    cfg_dir = Path(cfg_input_dir)
    if cfg_dir.exists():
        return cfg_dir

    workspace_input = Path(__file__).resolve().parents[2] / "input"
    if workspace_input.exists():
        return workspace_input

    cwd_input = Path.cwd() / "input"
    return cwd_input


def pad_to_square(image: Image.Image, fill: int | Tuple[int, int, int] = 0) -> Image.Image:
    """
    画像の長辺に合わせて正方形にパディングする。
    """
    w, h = image.size
    if w == h:
        return image

    max_side = max(w, h)
    new_img = Image.new(image.mode, (max_side, max_side), color=fill)

    paste_x = (max_side - w) // 2
    paste_y = (max_side - h) // 2
    new_img.paste(image, (paste_x, paste_y))
    return new_img


def compute_channel_mean_std(
    image_paths: List[Path],
    img_size: int,
    sample_limit: Optional[int] = None,
) -> tuple[List[float], List[float]]:
    """
    channel-wise mean/std 計算。
    長辺に合わせた正方形パディング → Resize → ToTensor で統計を取る。
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

    for p in tqdm(paths, desc="Compute mean/std (square padded)"):
        try:
            img = Image.open(p).convert("RGB")
            img = pad_to_square(img)
            x = tfm(img)
        except Exception:
            continue

        num_pix = x.shape[1] * x.shape[2]
        n += num_pix
        x_2d = x.reshape(3, -1)
        csum += x_2d.sum(dim=1).double()
        csum_sq += (x_2d ** 2).sum(dim=1).double()

    mean = (csum / n).tolist()
    var = (csum_sq / n - (csum / n) ** 2).tolist()
    std = [float(torch.sqrt(torch.tensor(max(v, 0.0))).item()) for v in var]
    return mean, std


class AtmaDataset(Dataset):
    def __init__(
        self,
        meta_df: pd.DataFrame,
        photos_dir: Path,
        img_size: int,
        mean: Optional[List[float]],
        std: Optional[List[float]],
        is_train: bool = True,
        task: str = "reg",
    ) -> None:
        self.meta_df = meta_df.reset_index(drop=True)
        self.photos_dir = Path(photos_dir)
        self.img_size = img_size
        self.is_train = is_train
        self.mean = mean
        self.std = std
        self.task = task

        tfms: List[torch.nn.Module] = []

        # 1. まず正方形にパディング（共通）
        tfms.append(pad_to_square)

        if is_train:
            # 学習時: RandomResizedCrop + Flip
            tfms.append(
                T.RandomResizedCrop(
                    size=img_size,
                    scale=(0.8, 1.0),
                    ratio=(0.75, 1.3333),
                )
            )
            tfms.append(T.RandomHorizontalFlip(p=0.5))
            tfms.append(T.RandomVerticalFlip(p=0.5))
        else:
            # 評価・推論時: 固定の Resize
            tfms.append(T.Resize((img_size, img_size)))

        # Tensor 化 + 正規化（共通）
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
            if self.task == "reg":
                label = float(row["target"])
                label_tensor = torch.tensor(label, dtype=torch.float32)
            else:
                label = int(row["target"])
                label_tensor = torch.tensor(label, dtype=torch.long)
        else:
            if self.task == "reg":
                label_tensor = torch.tensor(-1.0, dtype=torch.float32)
            else:
                label_tensor = torch.tensor(-1, dtype=torch.long)

        return img, label_tensor


# ==============================
# DINO backbone factory
# ==============================

def load_dino_checkpoint(path: Path):
    """
    PyTorch 2.6 以降の weights_only=True デフォルト変更に対応した loader。
    自分で学習した checkpoint を読む前提なので weights_only=False で読む。
    もし古い PyTorch で weights_only 引数が無い場合は、素直に fallback する。
    """
    try:
        # PyTorch 2.6 以降
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # 2.5 以前など weights_only 引数が無い場合
        return torch.load(path, map_location="cpu")


def create_dino_backbone(cfg: ExpConfig) -> tuple[nn.Module, int]:
    dino_dir = Path(cfg.dino_repo_dir).resolve()

    # ★ ここを追加：既に読み込まれている自前の utils をいったん消す
    if "utils" in sys.modules:
        del sys.modules["utils"]

    # ★ ここも修正：末尾ではなく先頭に入れる
    if str(dino_dir) not in sys.path:
        sys.path.insert(0, str(dino_dir))

    import vision_transformer as vits  # type: ignore

    if cfg.backbone_name == "vit_small":
        backbone = vits.vit_small(
            patch_size=cfg.dino_patch_size,
            num_classes=0,
        )
        embed_dim = cfg.dino_embed_dim_small
    elif cfg.backbone_name == "vit_base":
        backbone = vits.vit_base(
            patch_size=cfg.dino_patch_size,
            num_classes=0,
        )
        embed_dim = cfg.dino_embed_dim_base
    else:
        raise ValueError(f"Unsupported DINO backbone: {cfg.backbone_name}")

    ckpt_path = Path(cfg.dino_checkpoint_path)
    state_dict = load_dino_checkpoint(ckpt_path)
    key = cfg.dino_checkpoint_key
    if isinstance(state_dict, dict) and key in state_dict:
        state_dict = state_dict[key]

    cleaned = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "").replace("backbone.", "")
        cleaned[k] = v

    msg = backbone.load_state_dict(cleaned, strict=False)
    print(f"[DINO] Loaded weights from {ckpt_path}, msg: {msg}")

    return backbone, embed_dim


def create_dino_resnet_backbone(cfg: ExpConfig) -> Tuple[nn.Module, int]:
    """
    ResNet 系の DINO pretrain 重みを読み込む場合の factory。
    """
    model = timm.create_model("resnet50", pretrained=False, in_chans=3)
    ckpt_path = Path(cfg.dino_checkpoint_path)
    state_dict = load_dino_checkpoint(ckpt_path)
    key = cfg.dino_checkpoint_key
    if isinstance(state_dict, dict) and key in state_dict:
        state_dict = state_dict[key]

    cleaned = {k.replace("module.", "").replace("backbone.", ""): v
               for k, v in state_dict.items()}
    msg = model.load_state_dict(cleaned, strict=False)
    print(f"[DINO-ResNet] Loaded: {msg}")
    n_features = model.fc.in_features
    return model, n_features


# ==============================
# EMA
# ==============================

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}

        # 初期化時はとりあえず param と同じデバイスでクローンする
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if name not in self.shadow or not param.requires_grad:
                continue

            # ここで shadow を param と同じデバイスに移す
            shadow = self.shadow[name]
            if shadow.device != param.device:
                shadow = shadow.to(param.device)

            new_average = (1.0 - self.decay) * \
                param.detach() + self.decay * shadow
            # 計算結果は param 側のデバイスに乗っているので、そのまま保持
            self.shadow[name] = new_average.detach().clone()

    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                shadow = self.shadow[name]
                # param と shadow のデバイスをそろえる
                if shadow.device != param.device:
                    shadow = shadow.to(param.device)

                # 元のパラメータをバックアップしてから shadow を適用
                self.backup[name] = param.detach().clone()
                param.data.copy_(shadow.data)

                # shadow 自体も最新状態を保持
                self.shadow[name] = shadow

    def restore(self, model: nn.Module):
        # backup に保存しておいたオリジナルのパラメータを戻す
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name].data)
        self.backup = {}


# ==============================
# LightningModule
# ==============================


class AtmaLightningModule(pl.LightningModule):
    def __init__(
        self,
        exp_cfg: ExpConfig,
        learning_rate: float,
        weight_decay: float,
    ):
        super().__init__()
        self.exp_cfg = exp_cfg
        self.save_hyperparameters(ignore=["exp_cfg"])

        # --- backbone 作成 ---
        if exp_cfg.model_family == "resnet":
            self.backbone = timm.create_model(
                exp_cfg.backbone_name,
                pretrained=False,
                num_classes=0,
                in_chans=3,
            )
            feat_dim = self.backbone.num_features
        elif exp_cfg.model_family == "dino":
            if "resnet" in exp_cfg.backbone_name:
                self.backbone, feat_dim = create_dino_resnet_backbone(exp_cfg)
            else:
                self.backbone, feat_dim = create_dino_backbone(exp_cfg)
        else:
            raise ValueError(f"Unknown model_family: {exp_cfg.model_family}")

        if exp_cfg.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # --- head 作成 ---
        if exp_cfg.task == "reg":
            self.head = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, 1),
            )
            self.loss_fn = nn.MSELoss()
        elif exp_cfg.task == "cls":
            self.head = nn.Linear(feat_dim, exp_cfg.num_classes)
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown task: {exp_cfg.task}")

        # --- EMA 準備 ---
        self.use_ema = exp_cfg.use_ema
        if self.use_ema:
            self.ema = EMA(self, decay=exp_cfg.ema_decay)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """
        Lightning が checkpoint を保存するときに呼ばれるフック。
        EMA の shadow を一緒に保存する。
        """
        if self.use_ema:
            # CPU に移して保存しておくと後で扱いやすい
            ema_shadow_cpu = {
                name: param.detach().cpu()
                for name, param in self.ema.shadow.items()
            }
            checkpoint["ema_shadow"] = ema_shadow_cpu

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """
        Lightning が checkpoint を読み込んだ直後に呼ばれるフック。
        保存されていた EMA shadow を self.ema.shadow に戻す。
        """
        if self.use_ema and "ema_shadow" in checkpoint:
            ema_shadow = checkpoint["ema_shadow"]
            device = next(self.parameters()).device

            # パラメータのデバイスに合わせて shadow を復元
            restored_shadow = {}
            for name, tensor in ema_shadow.items():
                restored_shadow[name] = tensor.to(device)
            self.ema.shadow = restored_shadow

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        if isinstance(feat, tuple):
            feat = feat[0]
        out = self.head(feat)
        if self.exp_cfg.task == "reg":
            out = out.squeeze(1)
        return out

    # -------- Lightning hooks --------

    def training_step(self, batch, batch_idx: int):
        imgs, targets = batch
        preds = self(imgs)

        if self.exp_cfg.task == "reg":
            loss = self.loss_fn(preds, targets)
        else:
            loss = self.loss_fn(preds, targets.long())

        self.log("train_loss", loss, prog_bar=False,
                 on_step=False, on_epoch=True)
        return loss

    def on_after_backward(self):
        if self.use_ema:
            self.ema.update(self)

    def validation_step(self, batch, batch_idx: int):
        imgs, targets = batch

        if self.use_ema:
            self.ema.apply_shadow(self)
        preds = self(imgs)
        if self.use_ema:
            self.ema.restore(self)

        if self.exp_cfg.task == "reg":
            loss = self.loss_fn(preds, targets)
            rmse = torch.sqrt(loss)
            self.log("val_loss", loss, prog_bar=True,
                     on_step=False, on_epoch=True)
            self.log("val_rmse", rmse, prog_bar=True,
                     on_step=False, on_epoch=True)
            return {"val_loss": loss, "val_rmse": rmse}
        else:
            loss = self.loss_fn(preds, targets.long())
            acc = (preds.argmax(dim=1) == targets.long()).float().mean()
            self.log("val_loss", loss, prog_bar=True,
                     on_step=False, on_epoch=True)
            self.log("val_acc", acc, prog_bar=True,
                     on_step=False, on_epoch=True)
            return {"val_loss": loss, "val_acc": acc}

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        imgs, _ = batch
        if self.use_ema:
            self.ema.apply_shadow(self)
        preds = self(imgs)
        if self.use_ema:
            self.ema.restore(self)
        return preds

    def configure_optimizers(self):
        params = list(self.head.parameters())
        if not self.exp_cfg.freeze_backbone:
            params += list(self.backbone.parameters())

        optimizer = optim.Adam(
            params,
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
    を返す。
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
        task=cfg.exp.task,
    )
    val_ds = AtmaDataset(
        meta_df=df_val,
        photos_dir=photos_dir,
        img_size=cfg.exp.img_size,
        mean=img_mean,
        std=img_std,
        is_train=False,
        task=cfg.exp.task,
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

    module = AtmaLightningModule(
        exp_cfg=cfg.exp,
        learning_rate=cfg.exp.learning_rate,
        weight_decay=cfg.exp.weight_decay,
    )

    fold_out_dir = output_dir / f"fold{fold}"
    fold_out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        dirpath=str(fold_out_dir),
        filename="model-best",
        monitor="val_rmse" if cfg.exp.task == "reg" else "val_acc",
        mode="min" if cfg.exp.task == "reg" else "max",
        save_top_k=1,
        save_weights_only=True,
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")

    if cfg.exp.debug:
        os.environ["WANDB_MODE"] = "disabled"
    wandb_logger = WandbLogger(
        project=WANDB_PROJECT_NAME,
        name=f"{EXP_SHORT_NAME}_fold{fold}",
        group=EXP_SHORT_NAME,
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
    if cfg.exp.task == "reg":
        rmse = float(np.sqrt(np.mean((preds - y_true) ** 2)))
    else:
        # 分類タスクの場合は 1 - accuracy のような形で RMSE もどきを出しておく
        y_pred_cls = np.round(preds).astype(np.int64)
        rmse = float(np.mean(y_pred_cls != y_true))

    LOGGER.info("Fold %d RMSE (from best checkpoint): %.4f", fold, rmse)

    wandb.finish()
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
        task=cfg.exp.task,
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
        module = AtmaLightningModule.load_from_checkpoint(
            checkpoint_path=str(ckpt_path),
            exp_cfg=cfg.exp,
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

    # ========== mean/std 計算 ==========
    image_paths = [photos_dir / f"{oid}.jpg"
                   for oid in train_df["object_id"].tolist()]

    with trace("compute_mean_std"):
        img_mean, img_std = compute_channel_mean_std(
            image_paths=image_paths,
            img_size=cfg.exp.img_size,
            sample_limit=cfg.exp.mean_std_sample_limit,
        )

    LOGGER.info("Computed mean: %s", img_mean)
    LOGGER.info("Computed std : %s", img_std)

    # ========== StratifiedGroupKFold で fold 割り当て ==========
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

    # ========== fold 学習 & OOF 作成 ==========
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
    oof_path = output_dir / f"oof_{EXP_SHORT_NAME}_{len(used_folds)}folds.csv"
    oof_df.to_csv(oof_path, index=False)
    LOGGER.info("Saved OOF to %s", oof_path)

    # ====== Test 推論 + submission ======
    if not sample_sub_path.exists():
        LOGGER.error("Sample submission not found: %s", sample_sub_path)
        return

    submission_df = pd.read_csv(sample_sub_path)
    LOGGER.info(
        "Loaded sample submission: %s (shape=%s, columns=%s)",
        sample_sub_path,
        submission_df.shape,
        list(submission_df.columns),
    )

    # 元のカラム構成を控えておく
    orig_cols = list(submission_df.columns)

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

    LOGGER.info("Creating submission file from scratch using test_df...")

    # sample_submission.csv は使わず、test_df の ID を正とする
    submission_df = pd.DataFrame({

        "target": test_pred
    })

    # 保存
    sub_path = output_dir / \
        f"submission_{EXP_SHORT_NAME}_{len(fold_to_ckpt)}folds.csv"
    submission_df.to_csv(sub_path, index=False)

    LOGGER.info("Saved submission to %s", sub_path)
    LOGGER.info("Done.")


if __name__ == "__main__":
    main()
