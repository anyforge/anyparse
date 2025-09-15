from pathlib import Path
from PIL import Image
from omegaconf import OmegaConf
from .models.ocrbase import AnyOCRbase
from .models.ocrvllm import AnyOCRPlus


ROOT_PATH = Path(__file__).parent
DEFAULT_CONFIG_PATH = ROOT_PATH / "default_config.yaml"


class AnyOCR(object):
    def __init__(self, config_path):
        config_path = Path(config_path)
        if not config_path.exists():
            config_path = DEFAULT_CONFIG_PATH
        self.config = OmegaConf.load(config_path).anyocr
        if self.config.server == "mobile":
            self.ocr_config = self.config.ocr.mobile
        else:
            self.ocr_config = self.config.ocr.server 
        self.vllm_config = self.config.vllm
        self.ocr_base = AnyOCRbase(self.ocr_config)
        print(f"Loaded: AnyOCRbase-{self.config.server}")
        if self.config.type == "plus":
            self.ocr_vllm = AnyOCRPlus(self.vllm_config)
            print(f"Loaded: AnyOCRPlus")
        else:
            self.ocr_vllm = None

    def resize_image_if_need(self, image, max_dimension=1280):
        original_width, original_height = image.size
        # 检查是否需要缩放
        width_exceeds = original_width > max_dimension
        height_exceeds = original_height > max_dimension
        
        if not width_exceeds and not height_exceeds:
            return image
        # 计算缩放比例
        scale_factor = 1.0
        info = ''
        if width_exceeds and height_exceeds:
            # 两个都超过，取较小的缩放比例
            scale_factor = min(max_dimension / original_width, 
                            max_dimension / original_height)
            info = f"宽度和高度都超过{max_dimension}，缩放比例: {scale_factor:.3f}"
        elif width_exceeds:
            # 只有宽度超过
            scale_factor = max_dimension / original_width
            info = f"宽度超过{max_dimension}，缩放比例: {scale_factor:.3f}"
        elif height_exceeds:
            # 只有高度超过
            scale_factor = max_dimension / original_height
            info = f"高度超过{max_dimension}，缩放比例: {scale_factor:.3f}"
        
        # 计算新尺寸
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        
        # 使用高质量的重采样方法进行缩放
        resized_img = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        print(info)
        return resized_img
            
    def invoke_base(self, image: Image.Image, **kwargs):
        image = self.resize_image_if_need(image)
        output = self.ocr_base.invoke(image, **kwargs)
        return output
    
    def invoke_plus(self, image: Image.Image, **kwargs):
        image = self.resize_image_if_need(image)
        output = self.ocr_vllm.invoke(image, **kwargs)
        return output
    
    def invoke(self, image: Image.Image, mode = "plus", **kwargs):
        if mode == "plus":
            output = self.invoke_plus(image, **kwargs)
        else:
            output = self.invoke_base(image, **kwargs)
        return output
    
    async def ainvoke_stream(self, image: Image.Image, **kwargs):
        image = self.resize_image_if_need(image)
        output = self.ocr_vllm.ainvoke_stream(image, **kwargs)
        return output
            
            