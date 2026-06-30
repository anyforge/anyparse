import os
import json
import traceback
import datetime
import asyncio
from typing import Union,List,BinaryIO
from jinja2 import Environment as JinjaEnv
from openai import OpenAI,AsyncOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from .base import BaseASRClient
from ....loaders import load_audio_to_base64


class OpenAIClient(BaseASRClient):
    
    default_prompt_template = """
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": "{{ data_url }}"
                        }
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
    
    def _build_request_payload(
        self,
        audio: Union[str, os.PathLike, BinaryIO, bytes],
        max_new_tokens: int = 8192,
        **kwargs
    ):
        audio_dict = load_audio_to_base64(audio)
        audio_data = audio_dict["data"]
        audio_format = audio_dict["format"]
        audio_format = audio_format.lower() if audio_format.lower() != "mp3" else "mpeg"
        
        # img_data = self.encode_image(image)
        data_url = f"data:audio/{audio_format.lower()};base64,{audio_data}"
        # messages = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {
        #                 "type": "input_audio",
        #                 "input_audio": {
        #                     "data": data_url
        #                 }
        #             }
        #         ]
        #     }
        # ]
        messages = self.prompt_template.render(
            data_url = data_url
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
        audio: Union[str, os.PathLike, BinaryIO, bytes],
        max_new_tokens: int = 8192,
        **kwargs
    ) -> str:
        payload = self._build_request_payload(
            audio = audio,
            max_new_tokens = max_new_tokens,
            **kwargs
        )
        response = self.client.chat.completions.create(
            **payload
        )
        audio_text = response.choices[0].message.content
        return audio_text
    
    def run_batch(
        self, 
        batch_data, 
        max_new_tokens, 
        **kwargs
    ):
        task_timeout = kwargs.get("timeout", self.timeout) 
        results = [""] * len(batch_data)
        with ThreadPoolExecutor(max_workers=len(batch_data)) as executor:
            future_to_index = {}
            for idx, audio in enumerate(batch_data):           
                future = executor.submit(
                    self.invoke_one_process,
                    audio=audio,
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
        audio: Union[str, os.PathLike, BinaryIO, bytes],
        max_new_tokens: int = 8192,
        **kwargs
    ) -> str:
        payload = self._build_request_payload(
            audio = audio,
            max_new_tokens = max_new_tokens,
            **kwargs
        )
        response = await self.aclient.chat.completions.create(
            **payload
        )
        audio_text = response.choices[0].message.content
        return audio_text
        
    async def arun_batch(
        self, 
        batch_data, 
        max_new_tokens, 
        **kwargs
    ):
        """
        核心异步批处理逻辑
        负责将当前批次的数据转换为异步任务并并发执行
        """
        task_timeout = kwargs.get("timeout", self.timeout)
        tasks = []
        for audio in batch_data:
            coro = self.ainvoke_one_process(
                audio=audio,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
            task = asyncio.create_task(
                asyncio.wait_for(coro, timeout=task_timeout)
            )
            tasks.append(task)
        
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
        audios: List[str | os.PathLike | BinaryIO | bytes],
        max_new_tokens: int = 8192,
        batch_size: int = 1,
        **kwargs
    ) -> List[str]:
        if batch_size is None:
            batch_size = self.batch_size
        
        output = []
        
        # 2. 分批循环
        for i in range(0, len(audios), batch_size):
            batch_data = audios[i: i + batch_size]
            
            # 3. 在同步函数中运行异步逻辑
            # asyncio.run 会创建一个新的事件循环来运行协程，运行完后关闭
            batch_results = self.run_batch(batch_data, max_new_tokens, **kwargs)
            output.extend(batch_results)
            
        return output

    async def ainvoke(
        self,
        audios: List[str | os.PathLike | BinaryIO | bytes],
        max_new_tokens: int = 8192,
        batch_size: int = 1,
        **kwargs
    ) -> List[dict]:
        if batch_size is None:
            batch_size = self.batch_size
            
        output = []
        
        # 2. 分批循环
        for i in range(0, len(audios), batch_size):
            batch_data = audios[i: i + batch_size]
            
            # 3. 直接复用核心异步批处理方法
            batch_results = await self.arun_batch(batch_data, max_new_tokens, **kwargs)
            output.extend(batch_results)
            
        return output