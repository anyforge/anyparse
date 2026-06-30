import os
import time
import asyncio
import traceback
import pandas as pd
from .base import BaseConverter
from ..schemas import FileSnifferDetector
from ..utils.utils import clean_text_linebreak
from ..loggers import logger


class CsvConverter(BaseConverter):
    """
    csv转换器
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        
    def invoke_item(
        self, 
        file: str | os.PathLike,
        ftype: str = "csv",
        chunk_size: int = None,
        custom_separator: str = None,
        encoding: str = 'utf-8',
        auto_sniffer: bool = True,
        **kwargs
    ) -> list:
        try:
            parse_callback = kwargs.get("parse_callback")
            parse_callback.on_started(**{
                "file": file
            })
            if custom_separator is None:
                sep = ','
                if ftype == "tsv":
                    sep = '\t'
                elif ftype == "csv":
                    sep = ','

                if auto_sniffer:
                    sniffer_res = FileSnifferDetector.detect_file_delimiter(
                        file_path = file,
                        encoding = encoding
                    )
                    if sniffer_res.delimiter:
                        sep = sniffer_res.delimiter
            else:
                sep = custom_separator
            logger.debug(f"[auto sniffer: {auto_sniffer}, custom_separator: {custom_separator}, sniffer sep: {sep}]")
            res = []
            if chunk_size:
                chunk_size = int(chunk_size)
                dfset = pd.read_csv(
                    file,
                    sep = sep,
                    chunksize = chunk_size,
                    encoding = encoding
                )
                for idx,df in enumerate(dfset):
                    start_time = time.perf_counter()
                    df = df.dropna(how='all')
                    df = df.fillna("")
                    idx = idx + 1
                    content = df.to_markdown(index = False)
                    content = clean_text_linebreak(content)
                    res.append({
                        "sheet_name": "",
                        "chunk_id": f"{idx}",
                        "chunk_size": f"{chunk_size}",
                        "type": f"{ftype}",
                        "content": content,
                        "time_elapse": time.perf_counter() - start_time
                    })
            else:
                start_time = time.perf_counter()
                df = pd.read_csv(
                    file,
                    sep = sep,
                    encoding = encoding)
                df = df.dropna(how='all')
                df = df.fillna("")
                content = df.to_markdown(index = False)
                content = clean_text_linebreak(content)
                res.append({
                    "sheet_name": "",
                    "chunk_id": "1",
                    "chunk_size": None,
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
        ftype: str = "csv",
        chunk_size: int = None,
        custom_separator: str = None,
        encoding: str = 'utf-8',
        auto_sniffer: bool = True,
        **kwargs
    ) -> list:
        res = await asyncio.to_thread(
            self.invoke_item,
            file,
            ftype,
            chunk_size,
            custom_separator,
            encoding,
            auto_sniffer,
            **kwargs
        )
        return res