import gc
import cv2
import time
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List,Tuple
from transformers import AutoImageProcessor, AutoModelForImageClassification
from ....schemas import AnyDataModel
from ....exceptions import *


RAW_LABELS = ["0", "90", "180", "270"]


class OriClsOutput(AnyDataModel):
    elapse_times: float = 0.0
    score: float = 0.0
    label: str = RAW_LABELS[0]   


class DocOriClsModel:
    """
    旋转检测模型,支持batch运算
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
        self.model = AutoModelForImageClassification.from_pretrained(
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
    ) -> List[OriClsOutput]:
        if batch_size is None:
            batch_size = self.batch_size
        rotate_label = "0"  # Default to 0 if no rotation detected or not portrait
        output = []
        for i in range(0, len(images), batch_size):
            start_time = time.perf_counter()
            batch_data = images[i: i + batch_size]        
            inputs = self.image_processor(
                images = batch_data, 
                return_tensors = "pt"
            ).to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state
        
            probs = F.softmax(last_hidden_state, dim=-1)
            label_ids = torch.argmax(probs, dim=-1)
            scores = probs[torch.arange(probs.size(0)), label_ids]
            result = [[label.item(), score.item()] for label, score in zip(label_ids, scores)]  
            end_time = time.perf_counter()  
            for (predicted_label, score) in result:
                rotate_label = self.id2label[predicted_label]
                item = OriClsOutput(
                    label=rotate_label,
                    score=score,
                    elapse_times=end_time - start_time,
                )
                output.append(item) 
        return output

    def invoke_rotate(
        self, 
        images: List[Tuple[Image.Image, str]] | List[Tuple[np.ndarray, str]], 
    ) -> List:
        output = []
        for input_img,label in images:
            if isinstance(input_img, Image.Image):
                img = np.asarray(input_img)
            elif isinstance(input_img, np.ndarray):
                img = input_img
            else:
                raise AnyValueError("Input must be a pillow object or a numpy array.")
            if label == "270":
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif label == "90":
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif label == "180":
                img = cv2.rotate(img, cv2.ROTATE_180)
            else:
                # 0度不做处理
                pass
            output.append(img)
        return output