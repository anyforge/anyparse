import os
import io
import cv2
import json
import datetime
import asyncio
import base64
import traceback
from typing import Union,List,Tuple
import numpy as np
from PIL import Image
from jinja2 import Environment as JinjaEnv
from openai import OpenAI,AsyncOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from .base import BaseOCRModel
from ..otsls import convert_otsl_to_html,OTSL_FIND_PATTERN
from ....loaders import load_image_to_base64


class OpenAIClient(BaseOCRModel):
    
    default_prompt_template = """
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url", 
                        "image_url": {
                            "url": "{{ data_url }}"
                        }
                    },
                    {
                        "type": "text", 
                        "text": "{{ prompt }}"
                    }
                ]
            }
        ]
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        stream: bool = False,
        timeout: float = 1800.0,
        timeout_tolerance: float = 5.0,
        max_retries: int = 2,
        batch_size: int = 8,
        max_new_tokens: int = 8192,
        task_prompt_map: dict = {},
        prompt_template: str = "",
        t_patch_size: int = 2,
        min_pixels: int = 12544, # 112 * 112
        max_pixels: int = 71372800, # 14 * 14 * 4 * 1280
        image_format: str = "JPEG",
        patch_expand_factor: int = 1,
        **kwargs
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.kwargs = kwargs
        self.stream = stream
        self.timeout = timeout
        self.timeout_tolerance = timeout_tolerance
        self.max_retries = max_retries
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.task_prompt_map = task_prompt_map
        self.t_patch_size = t_patch_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.image_format = image_format
        self.patch_expand_factor = patch_expand_factor
        self.client_args = kwargs.get("client_args", {}) or {}
        self.call_args = kwargs.get("call_args", {}) or {}
        self._client = None
        self._aclient = None
        self._model_list = None
        self._amodel_list = None
        self.prompt_template = self.load_prompt_template(prompt_template)
        
    def load_prompt_template(self, prompt_template: str):
        prompt_template = prompt_template or self.default_prompt_template
        self.prompt_template = JinjaEnv().from_string(
            prompt_template
        )
        return self.prompt_template
        
    @property
    def client(self):
        if self._client is None:
            self._client = OpenAI(
                base_url = self.base_url,
                api_key = self.api_key,
                timeout = self.timeout,
                max_retries = self.max_retries,
                **self.client_args
            )
        return self._client
        
    @property
    def aclient(self):
        if self._aclient is None:
            self._aclient = AsyncOpenAI(
                base_url = self.base_url,
                api_key = self.api_key,
                timeout = self.timeout,
                max_retries = self.max_retries,
                **self.client_args
            )
        return self._aclient
    
    @property
    def model_list(self):
        if self._model_list is None:
            self._model_list = self.client.models.list().data
        return self._model_list
    
    @property
    async def amodel_list(self):
        if self._amodel_list is None:
            self._amodel_list = await self.aclient.models.list().data
        return self._amodel_list
        
    def encode_image(
        self, 
        image: Union[str, os.PathLike, Image.Image, np.ndarray]
    ):
        """将图片转为 Base64 字符串"""
        if isinstance(image, (str, os.PathLike)):
            # 确保路径存在
            if not os.path.exists(image):
                raise FileNotFoundError(f"图片文件不存在: {image}")
            
            with open(image, "rb") as image_file:
                img_bytes = image_file.read()
                img_data = base64.b64encode(img_bytes).decode("utf-8")
                
        elif isinstance(image, Image.Image):
            # 确保是 RGB 模式（防止 RGBA 或灰度图导致某些 API 报错，视具体模型而定）
            # 这里为了通用性，如果是 RGBA 转 RGB，如果是灰度图通常也能处理，按需调整
            if image.mode != "RGB":
                image = image.convert("RGB")       
            img_buffer = io.BytesIO()
            image.save(img_buffer, format="JPEG")
            img_bytes = img_buffer.getvalue()
            img_data = base64.b64encode(img_bytes).decode("utf-8")
            
        elif isinstance(image, np.ndarray):
            img_data = cv2.imencode('.jpg', image)[1]
            img_data = img_data.tostring()
            img_data = base64.b64encode(img_data).decode("utf-8")
            
        else:
            raise TypeError(f"不支持的图片类型: {type(image)}")             
        
        return img_data
    
    def get_prompts(self, task_name: str):        
        prompt_text = self.task_prompt_map.get(
            task_name, 
            "Text Recognition:"
        )
        return prompt_text

    def format_text(self, result_str: str, label_name: str):
        if label_name in ['doc_title','paragraph_title']:
            if not result_str.startswith('#'):
                result_str = f"# {result_str}"
                
        elif "formula" in label_name:
            if ("\\(" in result_str and "\\)" in result_str) or (
                "\\[" in result_str and "\\]" in result_str
            ):
                result_str = result_str.replace("$", "")
                result_str = (
                    result_str.replace("\\(", " $ ")
                    .replace("\\)", " $")
                    .replace("\\[\\[", "\\[")
                    .replace("\\]\\]", "\\]")
                    .replace("\\[", " $$ ")
                    .replace("\\]", " $$ ")
                )
                result_str = result_str.strip()
            if not result_str.startswith('$$'):
                result_str = f"$$\n{result_str}"
            if not result_str.endswith('$$'):
                result_str = f"{result_str}\n$$"
            if "formula_number" in label_name:
                if result_str.startswith("$") or result_str.endswith("$"):
                    result_str = result_str.replace("$", "")
                
        elif "table" in label_name:
            if OTSL_FIND_PATTERN.findall(result_str):
                result_str = convert_otsl_to_html(result_str)
        # elif label_name == "table":
        #     if result_str.startswith('<table') and result_str.endswith('/table>'):
        #         result_str = result_str
        #     else:
        #         result_str = f"<table>{result_str}</table>"
        return result_str
    
    def _build_request_payload(
        self, 
        image: Union[str, os.PathLike, Image.Image, np.ndarray], 
        task_name: str, 
        max_new_tokens: int = 8192, 
        **kwargs
    ) -> dict:
        img_data = load_image_to_base64(
            image,
            t_patch_size = kwargs.get("t_patch_size", self.t_patch_size),
            max_pixels = kwargs.get("max_pixels",self.max_pixels),
            image_format = kwargs.get("image_format",self.image_format),
            patch_expand_factor = kwargs.get("patch_expand_factor",self.patch_expand_factor),
            min_pixels = kwargs.get("min_pixels",self.min_pixels),
        )        
        data_url = f"data:image/{self.image_format.lower()};base64,{img_data}"
        prompt = self.get_prompts(task_name)
        # messages = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {
        #                 "type": "image_url", 
        #                 "image_url": {
        #                     "url": data_url
        #                 }
        #             },
        #             {
        #                 "type": "text",
        #                 "text": prompt
        #             }
        #         ]
        #     }
        # ]        
        messages = self.prompt_template.render(
            data_url = data_url,
            prompt = prompt
        )
        messages = json.loads(messages.strip())
        max_new_tokens = max_new_tokens if max_new_tokens <= self.max_new_tokens else self.max_new_tokens
        stream = kwargs.pop("stream", self.stream)
        call_args = self.call_args.copy()
        kwargs_call_args = kwargs.pop("call_args", {})
        call_args.update(kwargs_call_args)
        task_timeout = kwargs.pop("timeout", self.timeout) 
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "stream": stream,
            "timeout": task_timeout,
            **call_args
        }
        return payload
    
    def invoke_one_process(
        self,
        image: Union[str, os.PathLike, Image.Image, np.ndarray],
        task_name: str,
        max_new_tokens: int = 8192,
        **kwargs
    ) -> str:
        payload = self._build_request_payload(
            image = image,
            task_name = task_name,
            max_new_tokens = max_new_tokens,
            **kwargs
        )
        response = self.client.chat.completions.create(
            **payload
        )
        ocr_text = response.choices[0].message.content
        ocr_text = self.format_text(ocr_text, task_name)
        return ocr_text        
    
    def run_batch(
        self, 
        batch_data, 
        max_new_tokens, 
        **kwargs
    ) -> List[str]:
        task_timeout = kwargs.get("timeout", self.timeout) 
        results = [""] * len(batch_data)
        with ThreadPoolExecutor(max_workers=len(batch_data)) as executor:
            future_to_index = {}
            for idx, (image, task_map) in enumerate(batch_data):
                label_name = task_map["label"]            
                future = executor.submit(
                    self.invoke_one_process,
                    image = image,
                    task_name = label_name,
                    max_new_tokens = max_new_tokens,
                    **kwargs
                )
                future_to_index[future] = idx
            futures_done = as_completed(future_to_index, timeout=task_timeout)
            for future in futures_done:
                idx = future_to_index[future]
                try:
                    result = future.result(timeout=task_timeout)
                    results[idx] = result
                except Exception as e:
                    traceback.print_exc()
                    results[idx] = ""
        return results
        
    async def ainvoke_one_process(
        self,
        image: Union[str, os.PathLike, Image.Image, np.ndarray],
        task_name: str,
        max_new_tokens: int = 8192,
        **kwargs
    ) -> str:
        payload = self._build_request_payload(
            image = image,
            task_name = task_name,
            max_new_tokens = max_new_tokens,
            **kwargs
        )
        response = await self.aclient.chat.completions.create(
            **payload
        )
        ocr_text = response.choices[0].message.content
        ocr_text = self.format_text(ocr_text, task_name)
        return ocr_text
        
    async def arun_batch(
        self, 
        batch_data, 
        max_new_tokens, 
        **kwargs
    ) -> List[str]:
        """
        核心异步批处理逻辑
        负责将当前批次的数据转换为异步任务并并发执行
        """
        task_timeout = kwargs.get("timeout", self.timeout) 
        tasks = []
        for image, task_map in batch_data:
            label_name = task_map["label"]
            coro = self.ainvoke_one_process(
                image=image,
                task_name=label_name,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
            task = asyncio.create_task(
                asyncio.wait_for(coro, timeout=task_timeout)
            )
            tasks.append(task)
        # 核心改动：开启 return_exceptions=True
        # 这样即使某个任务超时或报错，其他任务依然会继续执行完毕
        try:
            processed_results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=task_timeout + self.timeout_tolerance # 预留一点缓冲时间
            )
        except asyncio.TimeoutError:
            # 如果整体超时，取消所有剩余任务
            for task in tasks:
                if not task.done():
                    task.cancel()
            # 返回对应数量的空字符串
            results = [""] * len(batch_data)
            return results
        
        results = []
        for i,result in enumerate(processed_results):
            if isinstance(result, Exception):
                print(f"[{datetime.datetime.now()}] Task {i} failed: {result}")
                results.append("")
            else:
                results.append(result)
        return results
        
    def invoke(
        self,
        images: List[Tuple[Image.Image, dict]],
        max_new_tokens: int = 8192,
        batch_size: int = 1,
        **kwargs
    ) -> List[str]:
        if batch_size is None or batch_size >= self.batch_size:
            batch_size = self.batch_size
        
        output = []
        
        # 2. 分批循环
        for i in range(0, len(images), batch_size):
            batch_data = images[i: i + batch_size]
            
            # 3. 在同步函数中运行异步逻辑
            # asyncio.run 会创建一个新的事件循环来运行协程，运行完后关闭
            batch_results = self.run_batch(batch_data, max_new_tokens, **kwargs)
            for item in batch_results:
                output.append(
                    {
                        "content": item,
                        "blocks": []
                    }
                )
            # output.extend(batch_results)
            
        return output

    async def ainvoke(
        self,
        images: List[Tuple[Image.Image, dict]],
        max_new_tokens: int = 8192,
        batch_size: int = 1,
        **kwargs
    ) -> List[str]:
        if batch_size is None or batch_size >= self.batch_size:
            batch_size = self.batch_size
            
        output = []
        
        # 2. 分批循环
        for i in range(0, len(images), batch_size):
            batch_data = images[i: i + batch_size]
            
            # 3. 直接复用核心异步批处理方法
            batch_results = await self.arun_batch(batch_data, max_new_tokens, **kwargs)
            for item in batch_results:
                output.append(
                    {
                        "content": item,
                        "blocks": []
                    }
                )
            # output.extend(batch_results)
            
        return output