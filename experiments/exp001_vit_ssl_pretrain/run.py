# experiments/exp001_vit_ssl_pretrain/run.py
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm.auto import tqdm
from pytorch_lightning.loggers import WandbLogger

import wandb
from utils.env import EnvConfig
from utils.logger import get_logger
from utils.timing import trace

LOGGER = None
WANDB_PROJECT_NAME = "atmacup11"
EXP_SHORT_NAME = Path(__file__).parent.name  # exp001_vit_ssl_pretrain


# ==============================
# Config
# ==============================

@dataclass
class ExpConfig:
    debug: bool = False
    seed: int = 42
    batch_size: int = 64
    epochs: int = 100
    img_size: int = 224
    num_workers: int = 4
    model_name: str = "vit_small_patch16_224"
    proj_dim: int = 256
    temperature: float = 0.1
    folds: List[int] = field(default_factory=lambda: [0])
    mean_std_sample_limit: Optional[int] = None


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    exp: ExpConfig = field(default_factory=ExpConfig)


cs = ConfigStore.instance()
cs.store(name="default", group="env", node=EnvConfig)
cs.store(name="default", group="exp", node=ExpConfig)


# ==============================
# Utility
# ==============================

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_input_dir(cfg_input_dir: str) -> Path:
    cfg_dir = Path(cfg_input_dir)
    if cfg_dir.exists():
        return cfg_dir

    workspace_input = Path(__file__).resolve().parents[2] / "input"
    if workspace_input.exists():
        return workspace_input

    cwd_input = Path.cwd() / "input"
    return cwd_input


def pad_to_square(image: Image.Image, fill: int | tuple[int, int, int] = 0) -> Image.Image:
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
    tfm = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
    ])

    n = 0
    csum = torch.zeros(3, dtype=torch.float64)
    csum_sq = torch.zeros(3, dtype=torch.float64)

    paths = image_paths[:sample_limit] if sample_limit is not None else image_paths

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


# ==============================
# Dataset for SSL
# ==============================

class SSLAtmaDataset(Dataset):
    """
    train + test の object_id を全部並べて、
    各サンプルについて「2 view の augment 画像」を返す Dataset。
    ラベルは使わない。
    """

    def __init__(
        self,
        object_ids: List[str],
        photos_dir: Path,
        img_size: int,
        mean: List[float],
        std: List[float],
    ) -> None:
        self.object_ids = object_ids
        self.photos_dir = Path(photos_dir)
        self.img_size = img_size
        self.mean = mean
        self.std = std

        base_tfm = [
            T.Lambda(pad_to_square),
            T.RandomResizedCrop(
                size=img_size,
                scale=(0.5, 1.0),
                ratio=(0.75, 1.3333),
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1,
            ),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std),
        ]
        self.transform = T.Compose(base_tfm)

    def __len__(self) -> int:
        return len(self.object_ids)

    def __getitem__(self, idx: int):
        oid = self.object_ids[idx]
        img_path = self.photos_dir / f"{oid}.jpg"
        img = Image.open(img_path).convert("RGB")

        v1 = self.transform(img)
        v2 = self.transform(img)

        return v1, v2


# ==============================
# Contrastive Loss (NT-Xent)
# ==============================

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    batch_size = z1.size(0)
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)

    reps = torch.cat([z1, z2], dim=0)  # (2N, D)
    sim_matrix = torch.matmul(reps, reps.T)  # (2N, 2N)

    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=reps.device)
    sim_matrix = sim_matrix[~mask].view(2 * batch_size, -1)

    positives = torch.sum(z1 * z2, dim=-1)  # (N,)
    positives = torch.cat([positives, positives], dim=0)  # (2N,)

    positives = positives / temperature
    negatives = sim_matrix / temperature

    labels = torch.zeros(2 * batch_size, dtype=torch.long, device=reps.device)
    logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)

    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss


# ==============================
# ViT Encoder + Projection Head
# ==============================

class ViTSSLModule(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        proj_dim: int,
        temperature: float,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.encoder = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            in_chans=3,
        )
        feat_dim = self.encoder.num_features

        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        z = self.projector(h)
        return z

    def training_step(self, batch, batch_idx: int):
        v1, v2 = batch
        z1 = self(v1)
        z2 = self(v2)
        loss = nt_xent_loss(z1, z2, temperature=self.hparams.temperature)
        self.log("train_ssl_loss", loss, prog_bar=True,
                 on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer


# ==============================
# main
# ==============================

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: Config) -> None:
    global LOGGER

    exp_dir_name = Path(sys.argv[0]).parent.name  # 例: exp001_vit_ssl_pretrain
    exp_name = f"{exp_dir_name}/default"
    base_output_dir = Path(cfg.env.exp_output_dir)
    output_dir = base_output_dir / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER = get_logger(__name__, output_dir)
    LOGGER.info("Start SSL pretrain: %s", exp_name)
    LOGGER.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    set_seed(cfg.exp.seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    if cfg.exp.debug:
        os.environ["WANDB_MODE"] = "disabled"

    input_dir = resolve_input_dir(cfg.env.input_dir)
    photos_dir = input_dir / "photos"
    train_csv = input_dir / "train.csv"
    test_csv = input_dir / "test.csv"

    LOGGER.info("Input dir: %s", input_dir)
    LOGGER.info("Train CSV: %s", train_csv)
    LOGGER.info("Test  CSV: %s", test_csv)
    LOGGER.info("Photos dir: %s", photos_dir)

    with trace("load_csv"):
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)

    train_df["object_id"] = train_df["object_id"].astype(str)
    test_df["object_id"] = test_df["object_id"].astype(str)

    all_object_ids = (
        train_df["object_id"].tolist() +
        test_df["object_id"].tolist()
    )
    LOGGER.info("Total images for SSL: %d", len(all_object_ids))

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

    ssl_ds = SSLAtmaDataset(
        object_ids=all_object_ids,
        photos_dir=photos_dir,
        img_size=cfg.exp.img_size,
        mean=img_mean,
        std=img_std,
    )

    ssl_loader = DataLoader(
        ssl_ds,
        batch_size=cfg.exp.batch_size,
        shuffle=True,
        num_workers=cfg.exp.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices: Optional[int] = 1 if torch.cuda.is_available() else None
    LOGGER.info("Using device: %s", device_str)

    module = ViTSSLModule(
        model_name=cfg.exp.model_name,
        proj_dim=cfg.exp.proj_dim,
        temperature=cfg.exp.temperature,
        learning_rate=1e-3,
        weight_decay=0.0,
    )

    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath=str(output_dir),
        filename="encoder_ssl",
        monitor="train_ssl_loss",
        mode="min",
        save_top_k=1,
        save_weights_only=True,
    )

    wandb_logger = WandbLogger(
        project=WANDB_PROJECT_NAME,
        name=f"{EXP_SHORT_NAME}_ssl",
        save_dir=str(output_dir),
    )

    trainer = pl.Trainer(
        max_epochs=cfg.exp.epochs,
        accelerator=accelerator,
        devices=devices,
        logger=wandb_logger,
        callbacks=[ckpt_cb],
        log_every_n_steps=10,
    )

    trainer.fit(module, train_dataloaders=ssl_loader)

    LOGGER.info("Best SSL checkpoint: %s", ckpt_cb.best_model_path)
    LOGGER.info("Done SSL pretrain.")


if __name__ == "__main__":
    main()
