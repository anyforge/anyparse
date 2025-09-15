import chardet
import concurrent.futures
from pathlib import Path
import pandas as pd
from typing import List, NamedTuple, Optional, Union, cast


class FileEncoding(NamedTuple):
    """File encoding as the NamedTuple."""

    encoding: Optional[str]
    """The encoding of the file."""
    confidence: float
    """The confidence of the encoding."""
    language: Optional[str]
    """The language of the file."""


def detect_file_encodings(file_path: Union[str, Path], file_type: str = 'text', timeout: int = 50) -> List[FileEncoding]:
    def read_and_detect(file_path: str) -> List[dict]:
        with open(file_path, "rb") as f:
            rawdata = f.read()
        encodings = cast(List[dict], chardet.detect_all(rawdata))
        if all(encoding["encoding"] is None for encoding in encodings):
            raise RuntimeError(f"Could not detect encoding for {file_path}")
        encodings = [FileEncoding(**enc) for enc in encodings if enc["encoding"] is not None]
        res = ""
        for encoding in encodings:
            try:
                if file_type == 'csv':
                    content = pd.read_csv(file_path,encoding=encoding.encoding)  
                else:
                    with open(file_path,'r',encoding=encoding.encoding) as f:
                        content = f.read()
                res = encoding.encoding
                del content
                break
            except UnicodeDecodeError:
                continue   
        return res         
        
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(read_and_detect, file_path)
        try:
            encodings = future.result(timeout=timeout)
        except Exception:
            encodings = ""
    return encodings
