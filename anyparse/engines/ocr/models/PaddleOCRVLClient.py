import asyncio
import torch
from PIL import Image
from pathlib import Path
from typing import List,Tuple,Union
from collections import Counter
from transformers import (
    AutoProcessor, 
    AutoModelForImageTextToText
)
from .base import BaseOCRModel
from ..otsls import convert_otsl_to_html,OTSL_FIND_PATTERN
from ....loaders import resize_image_if_need
from ....exceptions import *


def find_shortest_repeating_substring(s: str) -> Union[str, None]:
    """
    Find the shortest substring that repeats to form the entire string.

    Args:
        s (str): Input string.

    Returns:
        str or None: Shortest repeating substring, or None if not found.
    """
    n = len(s)
    for i in range(1, n // 2 + 1):
        if n % i == 0:
            substring = s[:i]
            if substring * (n // i) == s:
                return substring
    return None


def find_repeating_suffix(
    s: str, min_len: int = 8, min_repeats: int = 5
) -> Union[Tuple[str, str, int], None]:
    """
    Detect if string ends with a repeating phrase.

    Args:
        s (str): Input string.
        min_len (int): Minimum length of unit.
        min_repeats (int): Minimum repeat count.

    Returns:
        Tuple[str, str, int] or None: (prefix, unit, count) if found, else None.
    """
    for i in range(len(s) // (min_repeats), min_len - 1, -1):
        unit = s[-i:]
        if s.endswith(unit * min_repeats):
            count = 0
            temp_s = s
            while temp_s.endswith(unit):
                temp_s = temp_s[:-i]
                count += 1
            start_index = len(s) - (count * i)
            return s[:start_index], unit, count
    return None


def truncate_repetitive_content(
    content: str,
    line_threshold: int = 10,
    char_threshold: int = 10,
    min_len: int = 10,
    min_count: int = 3000,
) -> str:
    """
    Detect and truncate character-level, phrase-level, or line-level repetition in content.

    Args:
        content (str): Input text.
        line_threshold (int): Min lines for line-level truncation.
        char_threshold (int): Min repeats for char-level truncation.
        min_len (int): Min length for char-level check.

    Returns:
        Union[str, str]: (truncated_content, info_string)
    """
    if len(content) < min_count:
        return content

    stripped_content = content.strip()
    if not stripped_content:
        return content

    # Priority 1: Phrase-level suffix repetition in long single lines.
    if "\n" not in stripped_content and len(stripped_content) > 100:
        suffix_match = find_repeating_suffix(stripped_content, min_len=8, min_repeats=5)
        if suffix_match:
            prefix, repeating_unit, count = suffix_match
            if len(repeating_unit) * count > len(stripped_content) * 0.5:
                return prefix

    # Priority 2: Full-string character-level repetition (e.g., 'ababab')
    if "\n" not in stripped_content and len(stripped_content) > min_len:
        repeating_unit = find_shortest_repeating_substring(stripped_content)
        if repeating_unit:
            count = len(stripped_content) // len(repeating_unit)
            if count >= char_threshold:
                return repeating_unit

    # Priority 3: Line-level repetition (e.g., same line repeated many times)
    lines = [line.strip() for line in content.split("\n") if line.strip()]
    if not lines:
        return content
    total_lines = len(lines)
    if total_lines < line_threshold:
        return content
    line_counts = Counter(lines)
    most_common_line, count = line_counts.most_common(1)[0]
    if count >= line_threshold and (count / total_lines) >= 0.8:
        return most_common_line

    return content


class PPOCRVLClient(BaseOCRModel):
    def __init__(
        self,
        model_path: str = "resource/models/anyocr-vlm",
        batch_size: int = 1,
        max_new_tokens: int = 131072,
        dtype: str = "auto",
        device_map: str = "auto",
        attn_implementation=None, # "flash_attention_2"
        truncate_content: bool = True,
        truncate_content_list: list = [5000, 50],
        **kwargs
    ):
        self.truncate_content = truncate_content
        self.truncate_content_list = truncate_content_list
        self.kwargs = kwargs
        self.modelpath = Path(model_path).expanduser().resolve()
        if not self.modelpath.exists():
            raise AnyFileNotFoundError(
                f"{self.modelpath} not found!"
            )
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.dtype = dtype
        self.device_map = device_map
        self.attn_implementation = attn_implementation
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.modelpath, 
            dtype = self.dtype, 
            device_map = self.device_map,
            attn_implementation = self.attn_implementation
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            self.modelpath
        )        
        self.max_pixels = 1280 * 28 * 28
        self.min_pixels = self.processor.image_processor.size.shortest_edge
    
    def get_prompts(self, task_name: str):
        prompts = {
            "ocr": "OCR:",
            "table": "Table Recognition:",
            "formula": "Formula Recognition:",
            "chart": "Chart Recognition:",
            "spotting": "Spotting:",
            "seal": "Seal Recognition:",
        }
        task_name = task_name.lower()
        if "formula" in task_name:
            prompt_text = prompts["formula"]
        elif task_name == 'table':
            prompt_text = prompts["table"]
        elif task_name == 'chart':
            prompt_text = prompts["chart"]
        elif task_name == 'seal':
            prompt_text = prompts["seal"]
        else:
            prompt_text = prompts["ocr"]
        return prompt_text
    
    def format_text(self, result_str: str, label_name: str):
        if result_str is None:
            result_str = ""
        if self.truncate_content:
            min_count = self.truncate_content_list[0] if label_name == "table" else self.truncate_content_list[-1]
            result_str = truncate_repetitive_content(
                result_str, min_count=min_count
            )
        
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
    
    def generate(
        self, 
        image: Image.Image, 
        task_name: str,
        max_new_tokens: int = 8192,
        **kwargs
    ):
        max_new_tokens = max_new_tokens if max_new_tokens <= self.max_new_tokens else self.max_new_tokens
        prompt_text = self.get_prompts(task_name)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ]
            }
        ]
        
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            images_kwargs={
                "size": {
                    "shortest_edge": self.min_pixels, 
                    "longest_edge": self.max_pixels
                }
            },
        ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens
            )
        output_text = self.processor.decode(
            outputs[0][inputs["input_ids"].shape[-1]:-1]
        )
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
                        