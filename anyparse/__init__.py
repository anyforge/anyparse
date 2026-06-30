import importlib
from typing import TYPE_CHECKING

# modules class mapping
_LAZY_IMPORT_MAPPING = {
    # loggers
    "default_log_level": ".loggers",
    "anyparse_logger": ".loggers",
    "logger": ".loggers",
    
    # schemas
    "BaseModel": ".schemas",
    "ConfigDict": ".schemas",
    "Field": ".schemas",
    "AnyStrEnum": ".schemas",
    "AnyDataModel": ".schemas",
    "AnyConfig": ".schemas",
    "FileEncoding": ".schemas",
    "FileDelimiter": ".schemas",
    "FileSnifferDetector": ".schemas",
    "Metadata": ".schemas",
    "Page": ".schemas",
    "AnyParseOutput": ".schemas",
    "Element": ".schemas",
    "AnyOCROutput": ".schemas",
    "ASRTimeStamp": ".schemas",
    "AnyASROutput": ".schemas",
    "FileTypes": ".schemas",
    "MimeTypes": ".schemas",
    "Settings": ".schemas",
    
    # exceptions
    "AnyBaseError": ".exceptions",
    "AnyFileNotFoundError": ".exceptions",
    "AnyFileTypeError": ".exceptions",
    "AnyValueError": ".exceptions",
    "AnyRunTimeError": ".exceptions",
    
    # callbacks
    "BaseCallback": ".callbacks",
    "ParseCallback": ".callbacks",
    
    # engines.ocr
    "recursive_xy_cut": ".engines.ocr",
    "sorted_layout_boxes": ".engines.ocr",
    "DocOriClsModel": ".engines.ocr",
    "DocRectifierModel": ".engines.ocr",
    "BaseLayoutModel": ".engines.ocr",
    "LayoutConfig": ".engines.ocr",
    "AnyDocLayoutV3": ".engines.ocr",
    "BaseOCRClient": ".engines.ocr",
    "AnyOCR": ".engines.ocr",
    
    # parsers
    "BaseConverter": ".parsers",
    "ConverterClient": ".parsers",
    
    # pipelines
    "BaseParser": ".pipelines",
    "SyncExecutorMixin": ".pipelines",
    "AsyncExecutorMixin": ".pipelines",
    "AnyParser": ".pipelines",
    "AsyncAnyParser": ".pipelines",
    
    # apis
    "AnyParserApi": ".apis.app",
    
    # cli
    "BaseCLI": ".clis",
    "AnyParseCLI": ".clis",
    "anyparse_cli_main": ".clis",
    
    # version
    "__version__": ".__version__"
}


__all__ = list(_LAZY_IMPORT_MAPPING.keys())


# module __getattr__
def __getattr__(name: str):
    if name in _LAZY_IMPORT_MAPPING:
        module_path = _LAZY_IMPORT_MAPPING[name]
        # 动态导入对应的子模块
        module = importlib.import_module(module_path, package=__name__)
        # 从子模块中提取对应的类/对象
        attr = getattr(module, name)
        # 将其缓存到当前模块的全局变量中，避免下次重复 import
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# 4. [IDE 提示支持] 仅供静态类型检查器（如 mypy, PyCharm）使用
if TYPE_CHECKING:
    from .loggers import (
        default_log_level,
        anyparse_logger,
        logger
    )

    from .schemas import (
        BaseModel, ConfigDict, Field,
        AnyStrEnum,
        AnyDataModel,
        AnyConfig,
        FileEncoding,
        FileDelimiter,
        FileSnifferDetector,
        Metadata,
        Page,
        AnyParseOutput,
        Element,
        AnyOCROutput,
        ASRTimeStamp,
        AnyASROutput,
        FileTypes,
        MimeTypes,
        Settings
    )
    from .exceptions import (
        AnyBaseError,
        AnyFileNotFoundError,
        AnyFileTypeError,
        AnyValueError,
        AnyRunTimeError
    )
    from .callbacks import BaseCallback, ParseCallback
    from .engines.ocr import (
        recursive_xy_cut,
        sorted_layout_boxes,
        DocOriClsModel,
        DocRectifierModel,
        BaseLayoutModel,LayoutConfig,AnyDocLayoutV3,
        BaseOCRClient,AnyOCR
    )
    from .parsers import BaseConverter,ConverterClient
    from .pipelines import (
        BaseParser,
        SyncExecutorMixin,
        AsyncExecutorMixin,
        AnyParser,
        AsyncAnyParser
    )
    from .apis.app import AnyParserApi
    from .clis import BaseCLI, AnyParseCLI, anyparse_cli_main