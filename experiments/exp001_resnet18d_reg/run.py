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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

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
    # Dataset / DataLoader の作成
    # ==============================
    photos_dir = input_dir / "photos"
    LOGGER.info("Photos dir: %s", photos_dir)

    train_tfm, valid_tfm = create_transforms(cfg.exp.img_size)

    # とりあえず 1 つ目の fold を使う（config の exp.folds[0]）
    target_fold = cfg.exp.folds[0]
    LOGGER.info("Prepare DataLoader for fold=%d", target_fold)

    df_trn = train_df[train_df["fold"] != target_fold].copy()
    df_val = train_df[train_df["fold"] == target_fold].copy()

    LOGGER.info("Train size (fold != %d): %d", target_fold, len(df_trn))
    LOGGER.info("Valid size (fold == %d): %d", target_fold, len(df_val))

    trn_ds = AtmaDataset(meta_df=df_trn, images_root=photos_dir,
                         transform=train_tfm, is_train=True)
    val_ds = AtmaDataset(meta_df=df_val, images_root=photos_dir,
                         transform=valid_tfm, is_train=True)

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

    # 一つバッチを取り出して shape を確認（デバッグ目的）
    batch = next(iter(trn_loader))
    imgs, targets = batch
    LOGGER.info("Sample batch - imgs shape: %s, targets shape: %s",
                imgs.shape, targets.shape)

    LOGGER.info("Experiment finished (まだ Lightning での学習は未実装です)")


if __name__ == "__main__":
    main()
