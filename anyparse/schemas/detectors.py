import traceback
import chardet
import clevercsv
import concurrent.futures
from pathlib import Path
from typing import List, Iterable, Optional, Union, cast
from .base import AnyDataModel


class FileEncoding(AnyDataModel):
    encoding: Optional[str] = "utf-8"
    confidence: float = 0.0
    language: Optional[str] = ''


class FileDelimiter(AnyDataModel):
    delimiter: Optional[str] = ""
    quotechar: Optional[str] = ""
    escapechar: Optional[str] = ""
    strict: bool = False,    


class FileSnifferDetector(object):

    @staticmethod
    def detect_file_encodings(file_path: Union[str, Path], timeout: int = 10) -> FileEncoding:
        def read_and_detect(file_path: str) -> List[dict]:
            with open(file_path, "rb") as f:
                rawdata = f.read()
            encodings = cast(List[dict], chardet.detect_all(rawdata))
            if all(encoding["encoding"] is None for encoding in encodings):
                raise RuntimeError(f"Could not detect encoding for {file_path}")
            encodings = [FileEncoding(**enc) for enc in encodings if enc["encoding"] is not None]
            res = FileEncoding()
            for encoding in encodings:
                try:
                    with open(file_path,'r',encoding=encoding.encoding) as f:
                        for _ in range(5):
                            content = f.readline()
                    res = encoding
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
                encodings = FileEncoding()
        return encodings

    @staticmethod
    def detect_file_delimiter(
        file_path: Union[str, Path], 
        encoding: str,
        delimiters: Optional[Iterable[str]] = None
    ) -> FileDelimiter:
        try:
            with open(file_path, 'r', encoding = encoding) as f:
                lines = []
                for _ in range(10):
                    lines.append(f.readline())
                lines = ''.join(lines)
                dialect = clevercsv.Sniffer().sniff(lines, delimiters = delimiters, verbose = False)
                res = FileDelimiter(**dialect.to_dict())
        except:
            traceback.print_exc()
            res = FileDelimiter()
        finally:
            return res