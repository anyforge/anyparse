import os
import re
import sys
import json
import copy
import datetime
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Any, Dict, Set, Literal, List, Union, IO, Iterable
import yaml
try:
    import tomllib
    TOML_AVAILABLE = True
except ImportError:
    try:
        import toml as tomllib
        TOML_AVAILABLE = True
    except ImportError:
        TOML_AVAILABLE = False

if sys.version_info >= (3, 11):
    from enum import StrEnum as AnyStrEnum
else:
    from enum import Enum
    class AnyStrEnum(str, Enum):
        """
        A backport of Python 3.11+'s StrEnum for older versions.
        Ensures that the enum member behaves like a string in all contexts.
        """
        def __str__(self) -> str:
            return str(self.value)

        def __repr__(self) -> str:
            return f"{self.__class__.__name__}.{self.name}"

        # Optional: make it JSON-serializable without custom encoder
        def _generate_next_value_(name, start, count, last_values):
            return name.lower()


class AnyDataModel(BaseModel):
    model_config = ConfigDict(
        extra='allow',
        use_enum_values=True,
        arbitrary_types_allowed=True,
        # V2 中 json_encoders 的用法保持不变
        json_encoders={
            datetime.datetime: lambda v: v.isoformat(),
            datetime.date: lambda v: v.isoformat(),
            datetime.time: lambda v: v.isoformat(),
        }
    )
        
    def __repr__(self) -> str:
        return f'{self.__repr_name__()}({self.__repr_str__(", ")})'

    def __str__(self):
        return self.__repr__()
    
    def __getitem__(self, key: str):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"字段 '{key}' 不存在")
    
    def __setitem__(self, key: str, value: Any):
        setattr(self, key, value)
    
    def __delitem__(self, key: str):
        if hasattr(self, key):
            delattr(self, key)
        else:
            raise KeyError(f"字段 '{key}' 不存在")
    
    def __contains__(self, key: str):
        return hasattr(self, key)
    
    def __len__(self):
        return len(self.model_fields_set) + len(self.model_extra or {})
    
    def __iter__(self):
        return iter(self.keys())

    def __eq__(self, other: Any):
        if not isinstance(other, self.__class__):
            return False
        
        # 获取两个对象的所有字段数据
        self_data = self.model_dump()
        other_data = other.model_dump()
        
        # 移除 created 字段进行比较
        self_data.pop('created', None)
        other_data.pop('created', None)
        
        return self_data == other_data
    
    def __ne__(self, other: Any):
        return not self.__eq__(other)
    
    def __hash__(self):
        # 获取所有字段，但排除 created 字段
        data = self.model_dump()
        data.pop('created', None)  # 移除 created 字段
        
        # 只使用不可变字段计算哈希
        hashable_fields = []
        
        for field_name, field_value in data.items():
            if self._is_hashable(field_value):
                hashable_fields.append((field_name, field_value))
            else:
                # 对于不可哈希的字段，使用其字符串表示
                hashable_fields.append((field_name, str(field_value)))
        
        return hash(tuple(hashable_fields))
    
    def _is_hashable(self, obj: Any):
        try:
            hash(obj)
            return True
        except TypeError:
            return False
    
    def equals(self, other, 
               ignore_fields: Set[str] = None,
               ignore_created: bool = True):
        if not isinstance(other, self.__class__):
            return False
        
        # 准备忽略的字段集合
        ignore_set = set(ignore_fields or [])
        if ignore_created:
            ignore_set.add('created')
        
        # 获取数据
        self_data = self.model_dump()
        other_data = other.model_dump()
        
        # 移除忽略的字段
        for field in ignore_set:
            self_data.pop(field, None)
            other_data.pop(field, None)
        
        return self_data == other_data
    
    def keys(self):
        data = self.model_dump()
        return data.keys()
    
    def values(self):
        data = self.model_dump()
        return data.values()
    
    def items(self):
        data = self.model_dump()
        return data.items()
    
    def get(self, key: str, default: Any = None):
        try:
            return self[key]
        except KeyError:
            return default
    
    def pop(self, key: str, default: Any = None):
        try:
            value = self[key]
            del self[key]
            return value
        except KeyError:
            if default is None and len([arg for arg in [default] if arg is not None]) == 0:
                # 没有提供默认值参数
                raise
            return default
    
    def update(self, *args, **kwargs):
        for arg in args:
            if hasattr(arg, 'keys'):
                # 字典类型
                for key in arg:
                    self[key] = arg[key]
            else:
                # 可迭代对象，包含键值对
                for key, value in arg:
                    self[key] = value
        
        # 处理关键字参数
        for key, value in kwargs.items():
            self[key] = value
    
    def setdefault(self, key: str, default: Any = None):
        if key in self:
            return self[key]
        else:
            self[key] = default
            return default
    
    def clear(self):
        all_keys = list(self.keys())
        for key in all_keys:
            if key not in self.keys():
                try:
                    del self[key]
                except (KeyError, AttributeError):
                    pass
        
        for field_name, field_info in self.model_fields().items():
            if field_info.default is not None:
                self[field_name] = field_info.default
            elif field_info.default_factory is not None:
                self[field_name] = field_info.default_factory()
                
    # 增强的复制方法
    def copy(self, deep: bool = True, update: Dict[str, Any] = None):
        return self.model_copy(deep=deep, update=update)

    @classmethod
    def model_fields(cls):
        return getattr(cls, '__pydantic_fields__', {})

    def to_dict(self, mode: Literal['json', 'python'] | str = 'python', **kwargs):
        """转换为字典"""
        return self.model_dump(mode = mode, **kwargs)

    def to_json(self, **kwargs):
        """转换为JSON字符串"""
        return self.model_dump_json(**kwargs)

    def save_json(self, filename: str, encoding: str = 'utf-8', indent: int = 4, ensure_ascii: bool = False, **kwargs):
        """保存为 JSON 文件"""
        with open(filename, 'w', encoding=encoding) as f:
            json.dump(self.to_dict(mode = 'json'), f, indent = indent, ensure_ascii=ensure_ascii, **kwargs)
        return

    @classmethod
    def load_from_json(cls, filename: str, encoding: str = 'utf-8'):
        """从 JSON 文件加载"""
        with open(filename, 'r', encoding = encoding) as f:
            return cls.model_validate_json(f.read())

    @classmethod
    def load_from_dict(cls, data: Dict[str, Any]):
        """从字典加载"""
        return cls.model_validate(data)


class AnyConfig(object):
    # 环境变量解析正则表达式: ${oc.env:var_name, default_value}
    _ENV_PATTERN = re.compile(r'\$\{oc\.env:([^,}]+)(?:,\s*(.*?))?\}')
    # 配置变量引用正则表达式: ${var_name}
    _VAR_PATTERN = re.compile(r'\$\{([^}]+)\}')
    
    def __init__(
        self, 
        obj: dict = None,
        tags: Any = None,
        desc: str = ""
    ):
        """初始化 AnyConfig 对象
        
        Args:
            obj: 字典对象，会被递归转换为 AnyConfig
            tags: 标签
            desc: 描述信息
        """
        # 存储元数据
        object.__setattr__(self, '_tags', tags)
        object.__setattr__(self, '_desc', desc)
        object.__setattr__(self, '_config', {})
        
        if obj is not None and isinstance(obj, dict):
            self.create(obj)
            self.resolve_env()
    
    @classmethod
    def from_file(cls, file_: Union[str, Path, IO[Any]]) -> 'AnyConfig':
        """从json、yaml、toml文件创建配置对象
        
        Args:
            file_: 文件路径或者文件对象
            
        Returns:
            AnyConfig 对象
        """
        if isinstance(file_, (str, Path)):
            path = Path(file_)
            suffix = path.suffix.lower()
            with open(path, 'r', encoding='utf-8') as f:
                if suffix in ('.json'):
                    data = json.load(f)
                elif suffix in ('.yaml', '.yml'):
                    data = yaml.safe_load(f)
                elif suffix in ('.toml'):
                    if not TOML_AVAILABLE:
                        raise ImportError(
                            "TOML support requires tomllib (Python 3.11+) or toml package. "
                            "Install with: pip install toml"
                        )
                    data = tomllib.load(f)
                else:
                    raise ValueError(f"Unsupported file format: {suffix}, "
                                     f"only .json, .yaml, .yml, .toml supported")
        else:
            # 从文件对象读取，按顺序尝试
            data = None
            try:
                data = json.load(file_)
            except json.JSONDecodeError:
                file_.seek(0)
                try:
                    data = yaml.safe_load(file_)
                except yaml.YAMLError:
                    file_.seek(0)
                    if TOML_AVAILABLE:
                        try:
                            data = tomllib.load(file_)
                        except Exception:
                            pass
        
        if data is None:
            raise ValueError("Failed to parse config file, tried JSON, YAML, TOML")
        
        if not isinstance(data, dict):
            raise TypeError(f"Root of config file must be a dictionary, got {type(data)}")
            
        return cls(data)
    
    def create(self, obj: dict = None) -> 'AnyConfig':
        """将字典递归解析为 AnyConfig
        
        Args:
            obj: 输入字典
            
        Returns:
            self
        """
        object.__setattr__(self, '_config', {})
        if obj is None:
            obj = {}
        
        for key, value in obj.items():
            self._set_item_recursive(key, value)
        
        return self
    
    def _set_item_recursive(self, key: str, value: Any) -> None:
        """递归设置值，字典转为 AnyConfig，列表转为包含 AnyConfig 的列表"""
        if isinstance(value, dict):
            self._config[key] = AnyConfig(value)
        elif isinstance(value, list):
            self._config[key] = [
                AnyConfig(item) if isinstance(item, dict) 
                else self._process_list_item(item)
                for item in value
            ]
        else:
            self._config[key] = value
    
    def _process_list_item(self, item: Any) -> Any:
        """处理列表中的元素，递归转换字典"""
        if isinstance(item, dict):
            return AnyConfig(item)
        elif isinstance(item, list):
            return [
                AnyConfig(i) if isinstance(i, dict) 
                else self._process_list_item(i)
                for i in item
            ]
        else:
            return item
    
    def resolve_env(self) -> 'AnyConfig':
        """解析配置中的环境变量和配置变量引用
        - 环境变量格式: ${oc.env:var_name, default_value}
        - 配置变量引用: ${var_name} 引用当前配置中的其他变量
        - 支持相对路径: 在嵌套作用域中可以直接引用同级变量
        - 如果整个字符串就是一个引用，保持原始类型不转字符串
        - 迭代多次直到所有引用都解析完成
        
        如果环境变量存在则使用环境变量的值，否则使用默认值
        
        Returns:
            self
        """
        # 迭代解析，直到没有更多引用可以解析或者达到最大次数
        max_iterations = 10
        for _ in range(max_iterations):
            changed = self._resolve_once(self, self)
            if not changed:
                break
        return self
    
    def _resolve_once(self, current_scope: 'AnyConfig', root: 'AnyConfig') -> bool:
        """一次遍历解析，返回是否有变化"""
        changed = False
        for key, value in self._config.items():
            new_value = self._resolve_value(value, current_scope, root)
            if new_value is not value:
                self._config[key] = new_value
                changed = True
        return changed
    
    def _resolve_value(self, value: Any, current_scope: 'AnyConfig', root: 'AnyConfig') -> Any:
        """解析一个值中的所有引用，返回解析后的值"""
        if isinstance(value, str):
            return self._resolve_string(value, current_scope, root)
        elif isinstance(value, AnyConfig):
            # 递归解析子配置，当前作用域就是这个子配置
            changed = value._resolve_once(value, root)
            return value
        elif isinstance(value, list):
            # 递归解析列表
            for i, item in enumerate(value):
                new_item = self._resolve_value(item, current_scope, root)
                if new_item is not item:
                    value[i] = new_item
            return value
        elif isinstance(value, dict):
            # 递归解析字典
            for k, v in value.items():
                new_v = self._resolve_value(v, current_scope, root)
                if new_v is not v:
                    value[k] = new_v
            return value
        else:
            return value
    
    def _parse_value(self, value: str):
        """
        通用类型解析函数：
        尝试将字符串转换为最合适的 Python 类型 (int, float, bool, list, dict)
        """
        if not value:
            return value

        # 1. 尝试解析 JSON (支持 list 和 dict)
        # 只有当字符串以 [ 或 { 开头时才尝试，提高性能
        if value.startswith('[') or value.startswith('{'):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass # 不是合法的 JSON，继续向下执行

        # 2. 尝试布尔值
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'

        # 3. 尝试整数 (使用 isdigit 保持你原有的逻辑)
        if value.isdigit():
            return int(value)

        # 4. 尝试浮点数
        try:
            return float(value)
        except ValueError:
            pass

        # 5. 默认返回字符串
        return value
    
    def _resolve_string(self, text: str, current_scope: 'AnyConfig', root: 'AnyConfig') -> Any:
        """解析字符串中的所有引用
        
        如果整个字符串就是单个引用，返回原始值（保持类型）
        否则替换所有引用，返回字符串
        """
        # 检查是否整个字符串就是一个引用
        full_match = self._VAR_PATTERN.fullmatch(text)
        if full_match:
            var_name = full_match.group(1).strip()
            if not var_name.startswith('oc.env:'):
                value = self._find_var(var_name, current_scope, root)
                if value is not None:
                    return value  # 返回原始值，保持类型
        
        full_match_env = self._ENV_PATTERN.fullmatch(text)
        if full_match_env:
            var_name = full_match_env.group(1).strip()
            default = full_match_env.group(2)
            value = os.environ.get(var_name)
            if value is not None:
                # 尝试转换类型
                return self._parse_value(value)
            elif default is not None:
                default = default.strip()
                if (default.startswith('"') and default.endswith('"')) or \
                   (default.startswith("'") and default.endswith("'")):
                    default = default[1:-1]
                else:
                    # 尝试自动转换默认值类型
                    return self._parse_value(default)
                return default
            else:
                return ""
        
        # 多个引用或者部分引用，替换所有
        result = text
        
        # 先替换环境变量
        def replace_env(match: re.Match) -> str:
            var_name = match.group(1).strip()
            default = match.group(2)
            value = os.environ.get(var_name)
            if value is not None:
                return value
            elif default is not None:
                default = default.strip()
                if (default.startswith('"') and default.endswith('"')) or \
                   (default.startswith("'") and default.endswith("'")):
                    default = default[1:-1]
                return default
            else:
                return ""
        
        result = self._ENV_PATTERN.sub(replace_env, result)
        
        # 再替换配置变量引用
        def replace_var(match: re.Match) -> str:
            var_name = match.group(1).strip()
            if var_name.startswith('oc.env:'):
                return match.group(0)
            value = self._find_var(var_name, current_scope, root)
            if value is not None:
                return str(value)
            else:
                return match.group(0)
        
        result = self._VAR_PATTERN.sub(replace_var, result)
        return result
    
    def _find_var(self, var_path: str, current_scope: 'AnyConfig', root: 'AnyConfig') -> Any:
        """查找变量，先在当前作用域找，找不到再去根找
        
        支持:
        - simple_key - 当前作用域查找，找不到去根
        - parent.child - 点路径，从根开始查找
        """
        if '.' not in var_path:
            # 单个名称，先在当前作用域找
            if var_path in current_scope:
                return current_scope[var_path]
        # 点路径或者当前作用域找不到，去根找
        return root._get_var(var_path)
    
    def _get_var(self, var_path: str) -> Any:
        """根据点路径从当前获取变量值"""
        parts = var_path.split('.')
        current = self
        for part in parts:
            if isinstance(current, AnyConfig) and part in current:
                current = current[part]
            else:
                return None
        return current
    
    def merge(self, other: Union['AnyConfig', dict]) -> 'AnyConfig':
        """合并另一个配置对象，递归合并
        
        Args:
            other: 要合并的 AnyConfig 或者 dict
            
        Returns:
            新的合并后的 AnyConfig 对象
        """
        if isinstance(other, dict):
            other = AnyConfig(other)
        
        merged = AnyConfig(self.to_dict())
        merged._merge_recursive(merged._config, other._config)
        return merged
    
    def _merge_recursive(self, base: Dict[str, Any], update: Dict[str, Any]) -> None:
        """递归合并字典"""
        for key, value in update.items():
            if key in base and isinstance(base[key], AnyConfig) and isinstance(value, AnyConfig):
                # 如果两边都是 AnyConfig，递归合并
                self._merge_recursive(base[key]._config, value._config)
            else:
                # 否则直接覆盖
                base[key] = copy.deepcopy(value)
    
    def update(self, other: Union['AnyConfig', dict]) -> 'AnyConfig':
        """更新当前配置对象，就地修改
        
        Args:
            other: 要更新的配置
            
        Returns:
            self
        """
        if isinstance(other, dict):
            other = AnyConfig(other)
            
        self._merge_recursive(self._config, other._config)
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """将 AnyConfig 转换为字典，递归转换
        
        Returns:
            字典
        """
        result = {}
        for key, value in self._config.items():
            result[key] = self._to_dict_recursive(value)
        return result
    
    def _to_dict_recursive(self, value: Any) -> Any:
        """递归转换为字典"""
        if isinstance(value, AnyConfig):
            return value.to_dict()
        elif isinstance(value, list):
            return [self._to_dict_recursive(item) for item in value]
        else:
            return value
    
    def save_json(
        self, 
        file_: Union[str, Path, IO[Any]], 
        indent: int = 4, 
        ensure_ascii: bool = False, 
        **kwargs
    ) -> None:
        """保存为json文件
        
        Args:
            file_: 输出文件路径或文件对象
            indent: json缩进
            ensure_ascii: 是否确保ascii编码
        """
        data = self.to_dict()
        if isinstance(file_, (str, Path)):
            with open(file_, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, **kwargs)
        else:
            json.dump(data, file_, indent=indent, ensure_ascii=ensure_ascii, **kwargs)
    
    def save_yaml(
        self, 
        file_: Union[str, Path, IO[Any]],
        indent: int = 4,
        allow_unicode: bool = True,
        default_flow_style: bool = False,
        sort_keys: bool = False,
        **kwargs
    ) -> None:
        """保存为yaml文件
        
        Args:
            file_: 输出文件路径或文件对象
        """
        data = self.to_dict()
        if isinstance(file_, (str, Path)):
            with open(file_, 'w', encoding='utf-8') as f:
                yaml.dump(
                    data, 
                    f, 
                    indent = indent,
                    allow_unicode = allow_unicode, 
                    default_flow_style = default_flow_style,
                    sort_keys = sort_keys,
                    **kwargs
                )
        else:
            yaml.dump(
                data, 
                f, 
                indent = indent,
                allow_unicode = allow_unicode, 
                default_flow_style = default_flow_style,
                sort_keys = sort_keys,
                **kwargs
            )
    
    def save_toml(self, file_: Union[str, Path, IO[Any]], **kwargs) -> None:
        """保存为toml文件
        
        Args:
            file_: 输出文件路径或文件对象
        """
        if not TOML_AVAILABLE:
            raise ImportError(
                "TOML support requires tomllib (Python 3.11+) or toml package. "
                "Install with: pip install toml"
            )
        
        data = self.to_dict()
        if isinstance(file_, (str, Path)):
            with open(file_, 'w', encoding='utf-8') as f:
                tomllib.dump(data, f, **kwargs)
        else:
            tomllib.dump(data, file_, **kwargs)
    
    # ============ Python 魔法方法实现 ============
    
    def __getattr__(self, name: str) -> Any:
        """点访问: config.key"""
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        
        try:
            return self._config[name]
        except KeyError:
            raise AttributeError(f"'AnyConfig' object has no attribute '{name}'") from None
    
    def __setattr__(self, name: str, value: Any) -> None:
        """点赋值: config.key = value"""
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            self._set_item_recursive(name, value)
            self._config[name] = value
    
    def __delattr__(self, name: str) -> None:
        """删除属性: del config.key"""
        if name.startswith('_'):
            object.__delattr__(self, name)
        else:
            try:
                del self._config[name]
            except KeyError:
                raise AttributeError(f"'AnyConfig' object has no attribute '{name}'") from None
    
    def __getitem__(self, key: str) -> Any:
        """索引访问: config[key]"""
        return self._config[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """索引赋值: config[key] = value"""
        self._set_item_recursive(key, value)
        self._config[key] = value
    
    def __delitem__(self, key: str) -> None:
        """删除索引: del config[key]"""
        del self._config[key]
    
    def __contains__(self, key: str) -> bool:
        """in 操作: 'key' in config"""
        return key in self._config
    
    def __iter__(self) -> Iterable[str]:
        """迭代: for key in config"""
        return iter(self._config.keys())
    
    def __len__(self) -> int:
        """len: len(config)"""
        return len(self._config)
    
    def __bool__(self) -> bool:
        """布尔判断: if config"""
        return bool(self._config)
    
    def __eq__(self, other: Any) -> bool:
        """相等比较: config == other"""
        if not isinstance(other, AnyConfig):
            return False
        return self._config == other._config
    
    def __ne__(self, other: Any) -> bool:
        """不等比较: config != other"""
        return not self.__eq__(other)
    
    def __copy__(self) -> 'AnyConfig':
        """浅拷贝: copy.copy(config)"""
        return AnyConfig(self.to_dict())
    
    def __deepcopy__(self, memo: Dict[int, Any]) -> 'AnyConfig':
        """深拷贝: copy.deepcopy(config)"""
        return AnyConfig(copy.deepcopy(self.to_dict()))
    
    def __str__(self) -> str:
        """字符串表示: str(config)"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
    
    def __repr__(self) -> str:
        """repr 表示: repr(config)"""
        return f"AnyConfig({self.__str__()})"
    
    def copy(self) -> 'AnyConfig':
        """浅拷贝: config.copy()"""
        return self.__copy__()
    
    def deepcopy(self) -> 'AnyConfig':
        """深拷贝: config.deepcopy()"""
        return self.__deepcopy__(None)
    
    def keys(self):
        """返回所有键: dict 风格"""
        return self._config.keys()
    
    def values(self):
        """返回所有值: dict 风格"""
        return self._config.values()
    
    def items(self):
        """返回所有键值对: dict 风格"""
        return self._config.items()
    
    def get(self, key: str, default: Any = None) -> Any:
        """get 方法: dict 风格"""
        return self._config.get(key, default)
    
    def clear(self) -> None:
        """清空配置"""
        self._config.clear()
    
    def pop(self, key: str, default: Any = None) -> Any:
        """弹出元素"""
        if default is not None:
            return self._config.pop(key, default)
        else:
            return self._config.pop(key)
    
    # 让 pprint 能够正确打印
    def __dir__(self) -> List[str]:
        """自定义 dir() 输出，包含所有配置键"""
        attrs = list(super().__dir__())
        attrs.extend(self._config.keys())
        return sorted(attrs)
