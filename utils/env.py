# utils/env.py
from dataclasses import dataclass


@dataclass
class EnvConfig:
    """
    環境依存のパスをまとめて管理するクラス。
    デフォルトは「プロジェクト直下で実行する」ことを前提にした相対パス。
    """
    # 入力データのルート
    input_dir: str = "./input"
    # 出力のルート
    output_dir: str = "./output"
    # 実験ごとの出力をまとめる場所
    exp_output_dir: str = "./output/experiments"
