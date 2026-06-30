import os
import time
import asyncio
import traceback
from docutils.core import publish_parts
from bs4 import BeautifulSoup
from .base import BaseConverter
from .html import _CustomMarkdownify
from ..utils.utils import Readf, clean_text_linebreak
            
            
class MkdConverter(BaseConverter):
    """
    markdown转换器
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def invoke_item(
        self,
        file: str | os.PathLike,
        ftype: str = "md",
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
            if ftype == "md":
                content = []
                for chunk in Readf(file,encoding=encoding,strip = False):
                    content.append(chunk)
                content = ''.join(content)
            elif ftype == 'rst':
                with open(file, 'r', encoding=encoding) as f:
                    rst_text = f.read()
                html_parts = publish_parts(rst_text, writer_name='html')
                html_body = html_parts['html_body']
                soup = BeautifulSoup(html_body, "html.parser", from_encoding=encoding)
                for script in soup(["script", "style"]):
                    script.extract()

                # Print only the main content
                body_elm = soup.find("body")
                webpage_text = ""
                if body_elm:
                    webpage_text = _CustomMarkdownify(**kwargs).convert_soup(body_elm)
                else:
                    webpage_text = _CustomMarkdownify(**kwargs).convert_soup(soup)
                content = webpage_text.strip()             
            
            else:
                raise ValueError(f"ftype: {ftype} not supported!")
            
            content = clean_text_linebreak(content)
            
            res.append({
                "line_id": "1",
                "type": f"{ftype}",
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
        ftype: str = "md",
        encoding='utf-8', 
        **kwargs
    ) -> list:
        res = await asyncio.to_thread(
            self.invoke_item,
            file,
            ftype,
            encoding,
            **kwargs
        )
        return res