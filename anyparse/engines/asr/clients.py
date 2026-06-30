import os
import asyncio
from typing import Any,List,BinaryIO
from .base import BaseASRClient
from ...schemas import ASRTimeStamp,AnyASROutput
from ...utils.import_utils import import_module_resource
from ...exceptions import *
from ...loggers import logger


class AnyASR(BaseASRClient):
    def __init__(
        self,
        config: dict
    ):
        self.config = config
        self.audio_batch_size = self.config.get("audio_batch_size",1)
        self.asr_config = self.config['asr']
        self.asr_model_type = self.asr_config.get("model_type")
        if not self.asr_model_type:
            raise AnyValueError("asr_model_type is required")
        self.asr_model_config = self.asr_config.get(self.asr_model_type)
        if not self.asr_model_config:
            raise AnyValueError("asr_model_config is required")
        self.asr_model_class = self.asr_model_config.pop("model_class",None)
        if not self.asr_model_class:
            raise AnyValueError("asr_model_class is required")
        self.asr_model = import_module_resource(
                prefix_name = ".models",
                model_class = self.asr_model_class,
                package = __package__
            )
        self.asr_model = self.asr_model(**self.asr_model_config)
        logger.debug(f"Load model: {self.__class__.__name__}.{self.asr_model.__class__.__name__}")
        
    def invoke(
        self,
        audios: List[str | os.PathLike | BinaryIO | bytes],
        audio_batch_size: int = None,
        max_new_tokens: int = 16384,
        **kwargs
    ) -> List[AnyASROutput]:
        if audio_batch_size is None:
            audio_batch_size = self.audio_batch_size
        
        output = []
        for i in range(0, len(audios), audio_batch_size):
            batch_data = audios[i:i+audio_batch_size]
            batch_res = self.asr_model.invoke(
                audios = batch_data,
                max_new_tokens = max_new_tokens,
                batch_size = audio_batch_size,
                **kwargs
            )
            for item in batch_res:
                audio_text = item.get("text")
                item_res = AnyASROutput(
                    text = audio_text,
                    time_stamp = []
                )
                for line in item.get("time_stamp", []):
                    item_res.time_stamp.append(ASRTimeStamp(
                        start = line.get("start"),
                        end = line.get("end"),
                        text = line.get("text")
                    ))

        return output
    
    async def ainvoke(
        self,
        audios: List[str | os.PathLike | BinaryIO | bytes],
        max_new_tokens: int = 8192,
        audio_batch_size: int = 1,
        **kwargs
    ) -> List[AnyASROutput]:
        res = await asyncio.to_thread(
            self.invoke,
            audios,
            max_new_tokens,
            audio_batch_size,
            **kwargs
        )
        
        return res
    
    
