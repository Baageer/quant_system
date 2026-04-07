"""
装饰器工具模块
"""
import time
import functools
from typing import Callable, Any
from .logger import get_logger


logger = get_logger(__name__)


def timer(func: Callable) -> Callable:
    """计时装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} 执行耗时: {end_time - start_time:.4f}秒")
        return result
    return wrapper


def retry(max_attempts: int = 3, delay: float = 1.0):
    """重试装饰器"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts == max_attempts:
                        logger.error(f"{func.__name__} 执行失败，已重试{max_attempts}次: {e}")
                        raise
                    logger.warning(f"{func.__name__} 执行失败，{delay}秒后重试 (第{attempts}次): {e}")
                    time.sleep(delay)
        return wrapper
    return decorator


def log_execution(func: Callable) -> Callable:
    """执行日志装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"开始执行 {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"成功执行 {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"执行 {func.__name__} 失败: {e}")
            raise
    return wrapper


def validate_input(validator: Callable[[Any], bool], error_msg: str = "输入验证失败"):
    """输入验证装饰器"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for arg in args:
                if not validator(arg):
                    raise ValueError(error_msg)
            for value in kwargs.values():
                if not validator(value):
                    raise ValueError(error_msg)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def cache_result(func: Callable) -> Callable:
    """结果缓存装饰器"""
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    
    wrapper.cache_clear = lambda: cache.clear()
    return wrapper
