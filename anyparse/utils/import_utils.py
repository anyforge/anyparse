import importlib
import traceback
from typing import Any
from ..exceptions import *


def import_module_resource(
    prefix_name: str,
    model_class: str,
    package: str = None
) -> Any:
    try:
        module_path, class_name = model_class.rsplit('.', 1)
        module_name = f"{prefix_name}.{module_path}"
        # 核心修改：使用 f".models.{module_name}" 表示在当前包下的 models 文件夹里找
        # 并且必须传入 package=__package__ 告诉 Python 相对路径的起点在哪里
        target_class = getattr(
            importlib.import_module(
                name = f"{module_name}", 
                package = package
            ), 
            class_name
        )
        return target_class
    except ModuleNotFoundError:
        # 捕获模块找不到的情况（比如路径拼写错误、文件不存在）
        raise ImportError(f"动态导入失败：找不到模块 '{model_class}'，请检查路径是否正确。")
        
    except AttributeError:
        # 捕获模块内找不到指定类的情况（比如类名拼写错误）
        raise AttributeError(
            f"动态导入失败：在模块 '{module_name}' 中找不到类 '{class_name}'。\n"
        )
    
    except:
        traceback.print_exc()
        raise AnyValueError(f"{model_class} is not supported")