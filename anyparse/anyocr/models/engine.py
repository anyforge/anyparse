import os
import abc
import platform
import traceback
from enum import Enum
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
from typing import Any, Dict, Union, List, Sequence, Tuple
from onnxruntime import (
    GraphOptimizationLevel, InferenceSession, SessionOptions,
    get_available_providers, get_device
)
from .utils import (
    EngineType,
    ModelType,
    FileInfo
)

    
class EP(Enum):
    CPU_EP = "CPUExecutionProvider"
    CUDA_EP = "CUDAExecutionProvider"
    DIRECTML_EP = "DmlExecutionProvider"
    CANN_EP = "CANNExecutionProvider"


class ProviderConfig:
    def __init__(self, engine_cfg: Dict[str, Any]):
        self.had_providers: List[str] = get_available_providers()
        self.default_provider = self.had_providers[0]

        self.cfg_use_cuda = engine_cfg.get("use_cuda", False)
        self.cfg_use_dml = engine_cfg.get("use_dml", False)
        self.cfg_use_cann = engine_cfg.get("use_cann", False)

        self.cfg = engine_cfg

    def get_ep_list(self) -> List[Tuple[str, Dict[str, Any]]]:
        results = [(EP.CPU_EP.value, self.cpu_ep_cfg())]

        if self.is_cuda_available():
            results.insert(0, (EP.CUDA_EP.value, self.cuda_ep_cfg()))

        if self.is_dml_available():
            results.insert(0, (EP.DIRECTML_EP.value, self.dml_ep_cfg()))

        if self.is_cann_available():
            results.insert(0, (EP.CANN_EP.value, self.cann_ep_cfg()))

        return results

    def cpu_ep_cfg(self) -> Dict[str, Any]:
        return dict(self.cfg.cpu_ep_cfg)

    def cuda_ep_cfg(self) -> Dict[str, Any]:
        return dict(self.cfg.cuda_ep_cfg)

    def dml_ep_cfg(self) -> Dict[str, Any]:
        if self.cfg.dm_ep_cfg is not None:
            return self.cfg.dm_ep_cfg

        if self.is_cuda_available():
            return self.cuda_ep_cfg()
        return self.cpu_ep_cfg()

    def cann_ep_cfg(self) -> Dict[str, Any]:
        return dict(self.cfg.cann_ep_cfg)

    def verify_providers(self, session_providers: Sequence[str]):
        if not session_providers:
            raise ValueError("Session Providers is empty")

        first_provider = session_providers[0]

        providers_to_check = {
            EP.CUDA_EP: self.is_cuda_available,
            EP.DIRECTML_EP: self.is_dml_available,
            EP.CANN_EP: self.is_cann_available,
        }

        for ep, check_func in providers_to_check.items():
            if check_func() and first_provider != ep.value:
                print(
                    f"{ep.value} is available, but the inference part is automatically shifted to be executed under {first_provider}. "
                )

    def is_cuda_available(self) -> bool:
        if not self.cfg_use_cuda:
            return False

        CUDA_EP = EP.CUDA_EP.value
        if get_device() == "GPU" and CUDA_EP in self.had_providers:
            return True
        return False

    def is_dml_available(self) -> bool:
        if not self.cfg_use_dml:
            return False

        cur_os = platform.system()
        if cur_os != "Windows":
            return False

        window_build_number_str = platform.version().split(".")[-1]
        window_build_number = (
            int(window_build_number_str) if window_build_number_str.isdigit() else 0
        )
        if window_build_number < 18362:
            return False

        DML_EP = EP.DIRECTML_EP.value
        if DML_EP in self.had_providers:
            return True
        return False

    def is_cann_available(self) -> bool:
        if not self.cfg_use_cann:
            return False

        CANN_EP = EP.CANN_EP.value
        if CANN_EP in self.had_providers:
            return True
        return False


class InferSession(abc.ABC):

    @abc.abstractmethod
    def __init__(self, config):
        pass

    @abc.abstractmethod
    def __call__(self, input_content: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def _verify_model(model_path: Union[str, Path, None]):
        if model_path is None:
            raise ValueError("model_path is None!")

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} does not exists.")

        if not model_path.is_file():
            raise FileExistsError(f"{model_path} is not a file.")

    @abc.abstractmethod
    def have_key(self, key: str = "character") -> bool:
        pass

    @classmethod
    def get_model_url(cls, file_info: FileInfo) -> Dict[str, str]:
        model_dict = OmegaConf.select(
            cls.model_info,
            f"{file_info.engine_type.value}.{file_info.ocr_version.value}.{file_info.task_type.value}",
        )

        # 优先查找 server 模型
        if file_info.model_type == ModelType.SERVER:
            for k in model_dict:
                if (
                    k.startswith(file_info.lang_type.value)
                    and file_info.model_type.value in k
                ):
                    return model_dict[k]

        for k in model_dict:
            if k.startswith(file_info.lang_type.value):
                return model_dict[k]

        raise KeyError("File not found")

    @classmethod
    def get_dict_key_url(cls, file_info: FileInfo) -> str:
        model_dict = cls.get_model_url(file_info)
        return model_dict["dict_url"]
    

class OrtInferSession(InferSession):
    def __init__(self, cfg: Dict[str, Any]):
        # support custom session (PR #451)
        session = cfg.get("session", None)
        if session is not None:
            if not isinstance(session, InferenceSession):
                raise TypeError(
                    f"Expected session to be an InferenceSession, got {type(session)}"
                )

            self.session = session
            return

        model_path = cfg.get("model_path", None)
        if model_path is None:
            raise ValueError("model_path is None!")

        model_path = Path(model_path)
        self._verify_model(model_path)

        sess_opt = self._init_sess_opts(cfg.engine_cfg)

        provider_cfg = ProviderConfig(engine_cfg=cfg.engine_cfg)
        self.session = InferenceSession(
            model_path,
            sess_options=sess_opt,
            providers=provider_cfg.get_ep_list(),
        )
        provider_cfg.verify_providers(self.session.get_providers())

    @staticmethod
    def _init_sess_opts(cfg: Dict[str, Any]) -> SessionOptions:
        sess_opt = SessionOptions()
        sess_opt.log_severity_level = 4
        sess_opt.enable_cpu_mem_arena = cfg.enable_cpu_mem_arena
        sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        cpu_nums = os.cpu_count()
        intra_op_num_threads = cfg.get("intra_op_num_threads", -1)
        if intra_op_num_threads != -1 and 1 <= intra_op_num_threads <= cpu_nums:
            sess_opt.intra_op_num_threads = intra_op_num_threads

        inter_op_num_threads = cfg.get("inter_op_num_threads", -1)
        if inter_op_num_threads != -1 and 1 <= inter_op_num_threads <= cpu_nums:
            sess_opt.inter_op_num_threads = inter_op_num_threads

        return sess_opt

    def __call__(self, input_content: np.ndarray) -> np.ndarray:
        input_dict = dict(zip(self.get_input_names(), [input_content]))
        try:
            return self.session.run(self.get_output_names(), input_dict)[0]
        except Exception as e:
            error_info = traceback.format_exc()
            raise ONNXRuntimeError(error_info) from e

    def get_input_names(self) -> List[str]:
        return [v.name for v in self.session.get_inputs()]

    def get_output_names(self) -> List[str]:
        return [v.name for v in self.session.get_outputs()]

    def get_character_list(self, key: str = "character") -> List[str]:
        meta_dict = self.session.get_modelmeta().custom_metadata_map
        return meta_dict[key].splitlines()

    def have_key(self, key: str = "character") -> bool:
        meta_dict = self.session.get_modelmeta().custom_metadata_map
        if key in meta_dict.keys():
            return True
        return False


class ONNXRuntimeError(Exception):
    pass


def get_engine(engine_type: EngineType):
    if engine_type == EngineType.ONNXRUNTIME:
        return OrtInferSession
    raise ValueError(f"Unsupported engine: {engine_type.value}")