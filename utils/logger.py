# utils/logger.py
import logging
import time
from logging import INFO, FileHandler, StreamHandler
from pathlib import Path


def get_logger(file_name: str, file_dir: Path | str) -> logging.Logger:
    """
    コンソールとファイルの両方に出力する logger を返す。
    file_dir 配下に "YYYYmmdd_HHMMSS.log" というファイルを作る。
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(file_name)
    logger.setLevel(logging.INFO)

    # すでにハンドラが付いている場合は重複しないようにクリア
    if logger.handlers:
        logger.handlers.clear()

    # コンソール出力
    stream_handler = StreamHandler()
    stream_handler.setLevel(INFO)
    logger.addHandler(stream_handler)

    # ファイル出力
    file_dir = Path(file_dir)
    file_dir.mkdir(parents=True, exist_ok=True)
    log_path = file_dir / f"{time.strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = FileHandler(log_path)
    file_handler.setLevel(INFO)
    formatter = logging.Formatter(
        "[%(asctime)s : %(levelname)s - %(filename)s] %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
