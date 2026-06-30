import asyncio
import torch
from PIL import Image
from pathlib import Path
from typing import List,Tuple
from transformers import AutoProcessor, AutoModelForImageTextToText
from .base import BaseOCRModel
from ....loaders import load_image_to_base64, resize_image_if_need
from ....exceptions import *


class GLMOCRV1(BaseOCRModel):
    def __init__(
        self,
        model_path: str = "resource/models/anyocr-vlm",
        batch_size: int = 1,
        max_new_tokens: int = 8192,
        dtype: str = "auto",
        device_map: str = "auto",
        t_patch_size: int = 2,
        min_pixels: int = 12544, # 112 * 112
        max_pixels: int = 71372800, # 14 * 14 * 4 * 1280
        image_format: str = "JPEG",
        patch_expand_factor: int = 1,
        **kwargs
    ):
        self.modelpath = Path(model_path).expanduser().resolve()
        if not self.modelpath.exists():
            raise AnyFileNotFoundError(
                f"{self.modelpath} not found!"
            )
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.dtype = dtype
        self.device_map = device_map
        self.t_patch_size = t_patch_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.image_format = image_format
        self.patch_expand_factor = patch_expand_factor
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.modelpath, 
            dtype=self.dtype,
            device_map = self.device_map
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.modelpath)    

    def get_prompts(self, task_name: str):        
        prompts = {
            "text": "Text Recognition:",
            "formula": "Formula Recognition:",
            "table": "Table Recognition:"
        }  
        task_name = task_name.lower()
        if "formula" in task_name:
            prompt_text = prompts["formula"]
        elif 'table' in task_name:
            prompt_text = prompts["table"]
        else:
            prompt_text = prompts["text"]
        return prompt_text

    def format_text(self, result_str: str, label_name: str):
        if label_name in ['doc_title','paragraph_title']:
            if not result_str.startswith('#'):
                result_str = f"# {result_str}"
        elif "formula" in label_name:
            if not result_str.startswith('$$'):
                result_str = f"$$\n{result_str}"
            if not result_str.endswith('$$'):
                result_str = f"{result_str}\n$$"
        # elif label_name == "table":
        #     if result_str.startswith('<table') and result_str.endswith('/table>'):
        #         result_str = result_str
        #     else:
        #         result_str = f"<table>{result_str}</table>"
        return result_str
    
    def generate(
        self, 
        image: Image.Image, 
        task_name: str,
        max_new_tokens: int = 8192,
        **kwargs
    ):
        max_new_tokens = max_new_tokens if max_new_tokens <= self.max_new_tokens else self.max_new_tokens
        prompt_text = self.get_prompts(task_name)
        image_base64 = load_image_to_base64(
            image,
            t_patch_size = kwargs.get("t_patch_size", self.t_patch_size),
            max_pixels = kwargs.get("max_pixels",self.max_pixels),
            image_format = kwargs.get("image_format",self.image_format),
            patch_expand_factor = kwargs.get("patch_expand_factor",self.patch_expand_factor),
            min_pixels = kwargs.get("min_pixels",self.min_pixels),
        )
        data_url = f"data:image/{self.image_format.lower()};base64,{image_base64}"
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": data_url
                    },
                    {
                        "type": "text",
                        "text": prompt_text
                    }
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)
        inputs.pop("token_type_ids", None)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        output_text = self.processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=False
        )
        if output_text.endswith("<|user|>"):
            output_text = output_text[:-len("<|user|>")]
        return output_text
    
    def invoke(
        self, 
        images: List[Tuple[Image.Image, dict]],  
        # task_name: str = "text",
        use_image_resize: bool = False,
        max_new_tokens: int = 8192,
        batch_size: int = 1,
        **kwargs
    ) -> List[str]:
        if batch_size is None:
            batch_size = self.batch_size
            
        output = []
        for image,task_map in images:
            label_name = task_map["label"]
            if use_image_resize:
                image = resize_image_if_need(image)
            max_new_tokens = max_new_tokens if max_new_tokens <= self.max_new_tokens else self.max_new_tokens
            ocr_text = self.generate(image, label_name, max_new_tokens, **kwargs)
            ocr_text = self.format_text(ocr_text, label_name)
            ocr_text = {
                "content": ocr_text,
                "blocks": []
            }
            output.append(ocr_text)
        return output
    
    async def ainvoke(
        self, 
        images: List[Tuple[Image.Image, dict]],  
        # task_name: str = "text",
        use_image_resize: bool = False,
        max_new_tokens: int = 8192,
        batch_size: int = 1,
        **kwargs
    ) -> List[str]:
        res = await asyncio.to_thread(
            self.invoke,
            images,
            use_image_resize,
            max_new_tokens,
            batch_size,
            **kwargs
        )
        return res