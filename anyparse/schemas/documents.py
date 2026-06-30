from typing import List
from .base import AnyDataModel


class Metadata(AnyDataModel):
    """元数据"""
    file_idx: str = ""
    file_md5: str = ""
    file_type: str = ""
    file_name: str = ""
    file_size: str = ""
    version: str = ""
    
    
class Page(AnyDataModel):
    id: int = 0
    content: str = ""
    layout: List = []
    elapse_times: float = 0.0
    
    
class AnyParseOutput(AnyDataModel):
    metadata: Metadata = Metadata()
    pages: List[Page] = []
    content: str = ""
    elapse_times: float = 0.0
        

class Element(AnyDataModel):
    order_id: int = 0
    label: str = ""
    box: List = []
    content: str = ""
    

class AnyOCROutput(AnyDataModel):
    blocks: List[Element] = []
    
    
class ASRTimeStamp(AnyDataModel):
    start: float = 0.0
    end: float = 0.0
    text: str = ""

    
class AnyASROutput(AnyDataModel):
    text: str = ""
    time_stamp: List[ASRTimeStamp] = []