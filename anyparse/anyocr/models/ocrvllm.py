import time
import base64
import datetime
import traceback
from io import BytesIO
from PIL import Image
from threading import Thread
from pydantic import BaseModel
from transformers import (
    AutoTokenizer, 
    AutoProcessor, 
    AutoModelForImageTextToText, 
    TextIteratorStreamer,
    AsyncTextIteratorStreamer
)


class ocrVllmConfig(BaseModel):
    prompt: str = (
        """Extract the text from the above document as if you were reading it naturally. """ 
        """Return the tables in html format. Return the equations in LaTeX representation. """
        """If there is an image in the document and image caption is not present, """
        """add a small description of the image inside the <img></img> tag; """
        """otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. """
        """Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. """
        """Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""
    )
    

class AnyOCRPlusOutput(BaseModel):
    content: list = []
    elapse_times: dict = {}
    def __len__(self):
        if self.content is None:
            return 0
        return sum([len(x) for x in self.content])

    def to_markdown(self) -> str:
        return "\n".join(self.content)  
    

class AnyOCRPlusStream(BaseModel):
    content: str = ""
    elapse_times: dict = {}
    def __len__(self):
        if self.content is None:
            return 0
        return sum([len(x) for x in self.content])

    def to_markdown(self) -> str:
        return "\n".join(self.content)  
    

class AnyOCRPlus(object):
    def __init__(self, config):
        self.ocr_vllm_config = ocrVllmConfig()
        self.max_new_tokens = int(config.get("max_new_tokens", 8192))
        self.model_path = config.get("model_path", "resource/models/anyllm-ocr-s")
        self.torch_dtype = config.get("torch_dtype", "auto")
        self.device_map = config.get("device_map", "auto")
        # flash_attention_2
        self.attn_implementation = config.get("attn_implementation", None)
        self.vllm_model = AutoModelForImageTextToText.from_pretrained(
            self.model_path, 
            torch_dtype=self.torch_dtype, 
            device_map=self.device_map, 
            attn_implementation=self.attn_implementation
        )
        self.vllm_model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.prompt = self.ocr_vllm_config.prompt
        
    def pillow_to_base64(self, image):
        img_buffer = BytesIO()
        # image.save(img_buffer, format='JPEG')
        try:
            if image.mode == 'RGBA':
                image.save(img_buffer, format='PNG')
            else:
                image.save(img_buffer, format='JPEG')
        except:
            traceback.print_exc()
            image.save(img_buffer, format='PNG')
        byte_data = img_buffer.getvalue()
        base64_str = base64.b64encode(byte_data).decode("utf-8")
        return base64_str
        
    def invoke(self, 
               image: Image.Image, 
               prompt: str = "",
               max_new_tokens: int = 15000,
               **kwargs
        ):
        elapse_time = time.time()
        image_base64_str = self.pillow_to_base64(image)
        image_data = f"data:image/png;base64,{image_base64_str}"
        messages: list = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "image", "image": f"{image_data}"}, # f"file://{image_path}" or base64str
                {"type": "text", "text": prompt if prompt else self.prompt},
            ]},
        ]
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], 
            images=[image], 
            padding=True, 
            return_tensors="pt"
        )
        inputs = inputs.to(self.vllm_model.device)
        max_new_tokens=max_new_tokens if max_new_tokens > 0 else self.max_new_tokens
        output_ids = self.vllm_model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=False
        )
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]

        output_text = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )
        elapse_time = time.time() - elapse_time
        elapse_time = {
            "total": elapse_time
        }
        outputs = AnyOCRPlusOutput(
            content=output_text,
            elapse_times=elapse_time
        )
        return outputs
        
    async def ainvoke_stream(self, 
               image: Image.Image, 
               prompt: str = "",
               max_new_tokens: int = 15000,
               **kwargs
        ):
        elapse_time = time.time()
        image_base64_str = self.pillow_to_base64(image)
        image_data = f"data:image/png;base64,{image_base64_str}"
        messages: list = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "image", "image": f"{image_data}"}, # f"file://{image_path}" or base64str
                {"type": "text", "text": prompt if prompt else self.prompt},
            ]},
        ]
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], 
            images=[image], 
            padding=True, 
            return_tensors="pt"
        )
        inputs = inputs.to(self.vllm_model.device)
        streamer = AsyncTextIteratorStreamer(
            self.tokenizer,
            timeout=float(kwargs.get("timeout", 60.0)),
            skip_prompt=True,
            skip_special_tokens=True
        )
        max_new_tokens=max_new_tokens if max_new_tokens > 0 else self.max_new_tokens
        print("max_new_tokens::: ", max_new_tokens)
        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,  # Deterministic generation
            "pad_token_id": self.tokenizer.eos_token_id,
        }        
        # Start generation in a separate thread
        thread = Thread(target=self.vllm_model.generate,
                        kwargs=generation_kwargs)
        thread.start()
        # Yield generated tokens as they come
        try:
            async for new_text in streamer:
                # print(new_text, " ::: ", datetime.datetime.now())
                yield AnyOCRPlusStream(
                        content=new_text,
                        elapse_times={
                            "datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                    )            
        except:
            traceback.print_exc()

        finally:
            thread.join()
            
        elapse_time = time.time() - elapse_time
        elapse_time = {
            "total": elapse_time
        }
        outputs = AnyOCRPlusStream(
            content="",
            elapse_times=elapse_time
        )
        yield outputs