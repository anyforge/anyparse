import os
import asyncio
from abc import ABC,abstractmethod
from pathlib import Path
from typing import List
from ..loggers import logger
from ..schemas import (
    AnyConfig,
    Settings, FileTypes, MimeTypes,
    Metadata,AnyParseOutput,
    FileSnifferDetector
)
from ..callbacks import ParseCallback
from ..engines.ocr import AnyOCR
from ..parsers import ConverterClient
from ..exceptions import (
    AnyFileNotFoundError,AnyFileTypeError,
    AnyValueError,AnyRunTimeError
)
from ..utils.healths import AnyHealths
from ..utils.utils import autodlhash, format_filesize, create_uuid


class BaseParser(ABC):
    def _setup(self, config: dict | str | os.PathLike = {}):
        self.config = self.load_config(config)
        ocr_config = self.config['anyparse']
        filetypes_config = self.config['filetypes']
        mimetypes_config = self.config['mimetypes']
        self.settings = Settings(**ocr_config)
        self.filetypes_instance = FileTypes(**filetypes_config)
        self.mimetypes_instance = MimeTypes(**mimetypes_config)
        self.load_all_parsers(
            converters_models = self.settings.converters_models
        )
        self.load_all_models()
        self.load_all_healths()
        self.default_callback = ParseCallback()     
    
    @property
    def model_id(self) -> str:
        """获取当前使用的模型ID"""
        if self._model_id is None:
            self._model_id = "AnyParse"
        return self._model_id
    
    @property
    def version(self) -> str:
        """获取当前版本号"""
        if self._version_name is None:
            self._version_name = "1.0"
        return self._version_name       

    def load_config(
        self,
        config: dict | str | os.PathLike = {},
    ):
        if isinstance(config, dict):
            config = config
        elif isinstance(config, str | os.PathLike):    
            config = Path(config).expanduser().resolve()
            if not config.exists():
                raise AnyFileNotFoundError(
                    f"No such file or directory: {config.as_posix()}"
                )
            config = AnyConfig.from_file(config).to_dict()
        else:
            config = {}
        if not config:
            ocr_config = Settings().model_dump()
            filetypes_config = FileTypes().model_dump()
            mimetypes_config = MimeTypes().model_dump()
            config = {
                "filetypes": filetypes_config,
                "mimetypes": mimetypes_config,
                "anyparse": ocr_config
            }
        else:
            if (
                "anyparse" not in config or \
                "filetypes" not in config or \
                "mimetypes" not in config
            ):
                raise AnyValueError(
                    "Invalid config: missing key 'anyparse' or 'filetypes'"
                )
            else:
                ocr_config = Settings(**config["anyparse"]).model_dump()
                filetypes_config = FileTypes(**config["filetypes"]).model_dump()
                mimetypes_config = MimeTypes(**config["mimetypes"]).model_dump()
                config = {
                    "filetypes": filetypes_config,
                    "mimetypes": mimetypes_config,
                    "anyparse": ocr_config
                }                
        return config
    
    def load_all_parsers(
        self, 
        converters_models: List[dict], 
        **kwargs
    ):
        ### load parsers
        self.converters = ConverterClient(
            converters_models = converters_models,
            **kwargs
        )
        return    
    
    def load_all_models(self):
        self.ocr_model = AnyOCR(
            config = self.settings.model_dump()
        )
        return
    
    def load_all_healths(self):
        self.healths = AnyHealths()
    
    def check_health(self):
        flag = self.healths.check_health()
        return flag
    
    def build_parser_payload(
        self,
        file: str | os.PathLike = '', 
        metadata: Metadata = None,
        **kwargs
    ):
        realtime_args = self.settings.copy(deep = True, update = kwargs)
        kwargs = {x: y for x,y in kwargs.items() if x not in realtime_args.model_dump().keys()}
        if realtime_args.autodetect_encoding:
            encodings = self.detect_file_encodings(
                file,
                file_type = metadata.file_type
            )
            if encodings:
                realtime_args.text_encoding = encodings   
        return realtime_args,kwargs
    
    def build_document(
        self, 
        file: os.PathLike, 
        **kwargs
    ):
        autocal_md5 = kwargs.pop(
            "autocal_md5", 
            self.settings.autocal_md5
        )
        processfileidx = kwargs.pop(
            "file_idx", 
            None
        )
        processfileidx = create_uuid() if not processfileidx else processfileidx
        processfilename = file.name
        processfiletype = file.suffix.strip('.').lower()
        if processfiletype not in self.filetypes_instance.total_list():
            raise AnyFileTypeError(
                f"file type is not allow. file_type: {metadata.file_type}"
            )
            
        processfilehash = "" if not autocal_md5 else autodlhash().encrypt_file(file)
        processfilesize = format_filesize(file.stat().st_size)
        ### start process file
        metadata = Metadata(
            file_idx = processfileidx,
            file_md5 = processfilehash,
            file_type = processfiletype,
            file_name = processfilename,
            file_size = f"{processfilesize}",   
            version = f"{self.version.strip()}"   
        )       
        document = AnyParseOutput(
            metadata = metadata,
            pages = [],
            content = "",
            elapse_times = 0.0
        )    
        return document,metadata
    
    def detect_file_encodings(
        self,
        file: os.PathLike,
        file_type: str = "txt",
        default: str = "utf-8"
    ):
        text_encoding = default
        need_detect_files = [
            *self.filetypes_instance.text,
            *self.filetypes_instance.markdown,
            *self.filetypes_instance.html,
            *self.filetypes_instance.csv
        ]
        if file_type in need_detect_files:
            encodings = FileSnifferDetector.detect_file_encodings(
                file_path = file, 
                timeout = 15
            )
            if encodings:
                text_encoding = encodings.encoding
                logger.debug(f"detect_file_encodings: {text_encoding}")
        return text_encoding
    
    @abstractmethod
    def invoke(
        self, 
        file: str | os.PathLike = '', 
        **kwargs
    ):
        """
        解析文件。
        """
        pass
    
    @abstractmethod
    async def ainvoke(
        self, 
        file: str | os.PathLike = '', 
        **kwargs
    ):
        """
        异步封装 invoke 方法。
        使用 asyncio.to_thread 在独立线程中运行 CPU 密集型的解析任务，
        防止阻塞异步 Web 服务器的事件循环。
        """
        # 使用 to_thread 将同步阻塞的 invoke 移到线程池中执行
        # to_thread 在 Python 3.9+ 中可用
        return await asyncio.to_thread(self.invoke, file, **kwargs)


class SyncExecutorMixin(ABC):
    @abstractmethod
    def _execute(
        self, 
        file: str | os.PathLike = '', 
        **kwargs
    ):
        ...
    
    def invoke(
        self, 
        file: str | os.PathLike = '', 
        **kwargs
    ):
        """
        解析文件。
        """
        return self._execute(file, **kwargs)
        
    
    # 注意：异步方法在同步 Mixin 里直接报错，防止误用
    async def ainvoke(
        self, 
        file: str | os.PathLike = '', 
        **kwargs
    ):
        raise AnyRunTimeError("Cannot use async execute in sync.")


class AsyncExecutorMixin(ABC):
    @abstractmethod
    async def _aexecute(
        self, 
        file: str | os.PathLike = '', 
        **kwargs
    ):
        ...
    
    async def ainvoke(
        self, 
        file: str | os.PathLike = '', 
        **kwargs
    ):
        """
        解析文件。
        """
        return await self._aexecute(file, **kwargs)
    
    def invoke(
        self, 
        file: str | os.PathLike = '', 
        **kwargs
    ):
        raise AnyRunTimeError("Cannot use sync execute in async.")