"""
日志工具模块
"""
import logging
import logging.config
import yaml
import os
from typing import Optional


def setup_logger(
    config_path: str = "./config/logging.yaml",
    default_level: int = logging.INFO
) -> logging.Logger:
    """设置日志"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
    
    return logging.getLogger('quant_system')


def get_logger(name: str) -> logging.Logger:
    """获取日志记录器"""
    return logging.getLogger(name)


class LoggerMixin:
    """日志混入类"""
    
    @property
    def logger(self) -> logging.Logger:
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger
