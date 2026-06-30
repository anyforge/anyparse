import os
import time
import json
import asyncio
import traceback
from .base import BaseConverter
from ..utils.utils import clean_text_linebreak


class IpynbConverter(BaseConverter):
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
            res = []
            with open(file,'r',encoding = encoding) as f:
                nbd = json.loads(f.read())    
            title = nbd.get("metadata", {})
            for cell_idx, cell in enumerate(nbd.get("cells", [])):
                md_output = []
                start_time = time.perf_counter()
                cell_type = cell.get("cell_type", "")
                source_lines = cell.get("source", [])
                source_lines = "".join(source_lines)
                if not source_lines:
                    continue
                if cell_type == "markdown":
                    md_output.append(source_lines)

                elif cell_type == "code":
                    # Code cells are wrapped in Markdown code blocks
                    md_output.append(f"```python\n{source_lines}\n```")
                elif cell_type == "raw":
                    md_output.append(f"```\n{source_lines}\n```")

                content = "\n\n".join(md_output)
                content = clean_text_linebreak(content)
                res.append({
                    "cell_idx": cell_idx + 1,
                    "type": "text",
                    "content": content,
                    "title": title,
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