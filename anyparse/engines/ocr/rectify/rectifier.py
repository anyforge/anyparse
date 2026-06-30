import cv2
import time
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List
from transformers import AutoImageProcessor, AutoModel
from ....schemas import AnyDataModel
from ....exceptions import *


class DocRectifierOutput(AnyDataModel):
    image: np.ndarray = None
    elapse_times: float = 0.0


class DocRectifierModel:
    """
    旋转检测模型, 支持batch运算
    """
    def __init__(
        self,
        config: dict = {},
        **kwargs
    ):
        self.config = config
        self.kwargs = kwargs
        self.batch_size = self.config.get("batch_size", 1)
        self.dtype = self.config.get("dtype", "auto")
        self.device_map = self.config.get("device_map", "auto")
        self.model_path = Path(self.config["model_path"]).expanduser().resolve()
        if not self.model_path.exists():
            raise AnyFileNotFoundError(f"{self.model_path} not exists")
            # raise ValueError(f"{self.config.model_path} not exists")
        self.model = AutoModel.from_pretrained(
            self.model_path, 
            dtype=self.dtype,
            device_map = self.device_map
        ).eval()
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.model_path
        )  
        self.id2label = self.model.config.id2label

    def close(self):
        self.model = None
        self.image_processor = None

    def __del__(self):
        """析构函数：兜底清理（不保证立即执行）"""
        self.close()
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def invoke(
        self, 
        images: List[Image.Image],
        batch_size: int = 1,
    ) -> List[DocRectifierOutput]:
        if batch_size is None:
            batch_size = self.batch_size
            
        output = []
        for i in range(0, len(images), batch_size):
            batch_data = images[i: i + batch_size]
            start_time = time.perf_counter()
            inputs = self.image_processor(
                images=batch_data, 
                return_tensors="pt"
            ).to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            result = self.image_processor.post_process_document_rectification(
                outputs.last_hidden_state, 
                inputs["original_images"]
            )
            end_time = time.perf_counter()
            for rect_image in result:
                rect_image = rect_image['images'].detach().cpu().numpy()
                item = DocRectifierOutput(
                    image = rect_image,
                    elapse_times=end_time - start_time,
                )
                output.append(item)
        return output