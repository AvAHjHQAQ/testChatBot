"""
全双工语音对话系统 v2.0 - 日志模块
支持loguru回退到标准logging
"""
import logging

try:
    from loguru import logger
    HAS_LOGURU = True
except ImportError:
    HAS_LOGURU = False
    # 创建标准logging的logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    logger = logging.getLogger("voice_dialog")

__all__ = ["logger", "HAS_LOGURU"]
