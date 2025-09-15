from abc import ABC
from pydantic import BaseModel,Field
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    TypedDict,
    Union,
    cast,
)


class Serializable(BaseModel, ABC):
    """Serializable base class.
    """
    class Config:
        extra = 'allow'  # 允许额外的字段
        arbitrary_types_allowed = True #允许 Pydantic 处理任意类型

    # Remove default BaseModel init docstring.
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """"""
        super().__init__(*args, **kwargs)

    def __repr_args__(self) -> Any:
        return [
            (k, v)
            for k, v in super().__repr_args__()
        ]

    def to_json(self) -> Any:
        """Serialize the object to JSON.
        """
        _id: List[str] = []
        try:
            if hasattr(self, "__name__"):
                _id = [*self.__module__.split("."), self.__name__]
            elif hasattr(self, "__class__"):
                _id = [*self.__class__.__module__.split("."), self.__class__.__name__]
        except:
            pass
        kwargs = {
            k: getattr(self, k, v) for k, v in self
        }
        kwargs = {
            "type": "constructor",
            "id": _id,
            "kwargs": kwargs
        }
        return kwargs
    

class Metadata(Serializable):
    """元数据"""
    file_id: str = ""
    file_type: str = ""
    file_name: str = ""
    file_size: str = ""
    file_md5: str = ""
    

class Pagedata(Serializable):
    """页面内容"""
    pageid: str = ''
    meta: str = ''
    content: str = ''
    images: Dict = {}
    audios: Dict = {}
    videos: Dict = {}


class AnyParseOutput(Serializable):
    """文档内容"""
    metas: Metadata = Metadata()
    pages: List[Pagedata] = []
    elapse_times: dict = {}
  
    
class AnyParseStream(Serializable):
    """文档内容"""
    metas: Metadata = Metadata()
    pageid: str = ''
    content: str = ""
    elapse_times: dict = {}    
    