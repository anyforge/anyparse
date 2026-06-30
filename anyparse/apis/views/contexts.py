from fastapi import Depends
from pydantic import BaseModel
from pathlib import Path
from ...schemas.configs import (
    default_cache_dir,
    default_maxsize,
    default_file_read_size    
)


class KeyPrefix(BaseModel):
    task: str = "task"  


class _AppViewsContext:
    def __init__(self):
        self._instance = None

    def set_instance(self, instance):
        if self._instance is not None:
            raise RuntimeError("AppViews instance already set")
        self._instance = instance

    def get_instance(self):
        if self._instance is None:
            raise RuntimeError(
                "AppViews instance not initialized. "
                "Make sure create_app() has been called."
            )
        return self._instance


# 全局上下文单例
_appviews_ctx = _AppViewsContext()


def set_appviews_instance(instance):
    """供 appviews.create_app() 调用，注册实例"""
    _appviews_ctx.set_instance(instance)


def get_appviews_instance():
    """供 Depends 使用，获取 appviews 实例"""
    return _appviews_ctx.get_instance()


def get_appviews_config():
    """获取已存在的配置实例"""
    return _appviews_ctx.get_instance().api_config


def get_cache_dir():
    api_config = get_appviews_config()
    cache_dir = api_config.get("anyparse", {}).get("cache_dir", "")
    if not cache_dir:
        cache_dir = default_cache_dir
    cache_dir = Path(cache_dir).expanduser().resolve()
    return cache_dir


# 快捷依赖项（可选，语义更清晰）
mainviews = Depends(get_appviews_instance)