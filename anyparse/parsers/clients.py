from typing import List, Dict
from .base import BaseConverter
from ..utils.import_utils import import_module_resource
from ..loggers import logger


class ConverterClient(object):
    def __init__(self, converters_models: List[dict], **kwargs):
        self.kwargs = kwargs    
        self._converters: Dict[str, BaseConverter] = {}
        for item in converters_models:
            # {"name": "email", "model_class": "email.EmailConverter"}
            item_name = item['name']
            item_model_class = item['model_class']
            item_model = import_module_resource(
                prefix_name = "",
                model_class = item_model_class,
                package = __package__
            )
            self.register_converter(item_name, item_model)
        logger.debug(f"regist converters: {list(self._converters.keys())}")
        
    def register_converter(self, file_type: str, converter_class: BaseConverter):
        self._converters[file_type] = converter_class(**self.kwargs)
    
    def __dir__(self):
        # 1. 先获取父类（object）默认返回的所有属性和方法
        default_dir = list(super().__dir__())
        # 2. 获取当前 _converters 字典中注册的所有转换器名称（即字典的键）
        converter_names = list(self._converters.keys())
        # 3. 将两者合并，并返回一个去重后的列表
        return default_dir + converter_names
    
    def __getattr__(self, name):
        if name in list(self._converters.keys()):
            return self._converters[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __getitem__(self, name):
        if name in list(self._converters.keys()):
            return self._converters[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    