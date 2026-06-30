import os
import time
import asyncio
import traceback
from .base import BaseConverter
from ..utils.utils import Readf, clean_text_linebreak


class TextConverter(BaseConverter):
    """
    txt转换器
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def invoke_item(
        self,
        file: str | os.PathLike,
        encoding='utf-8', 
        **kwargs
    ) -> list:
        try:
            parse_callback = kwargs.get("parse_callback")
            parse_callback.on_started(**{
                "file": file
            })
            start_time = time.perf_counter()
            res = []
            content = []
            for chunk in Readf(file,encoding=encoding,strip = False):
                content.append(chunk)
            content = ''.join(content)
            content = clean_text_linebreak(content)
            res.append({
                "line_id": "1",
                "type": "text",
                "content": content,
                "time_elapse": time.perf_counter() - start_time
            })
            parse_callback.on_finished(**{
                "file": file
            })  
        except:
            traceback.print_exc()
            res = []
        finally:
            return res
        
    async def ainvoke_item(
        self,
        file: str | os.PathLike,
        encoding='utf-8', 
        **kwargs
    ) -> list:
        res = await asyncio.to_thread(
            self.invoke_item,
            file,
            encoding,
            **kwargs
        )
        return res