import asyncio
import traceback
from PIL import Image
from typing import Any,List,Tuple,Dict
from .base import BaseOCRClient
from .rectify.oricls import DocOriClsModel
from .rectify.rectifier import DocRectifierModel
from .layout import AnyDocLayoutV3
from ...schemas import Element, AnyOCROutput
from ...loaders import cv2_to_pillow
from ...utils.import_utils import import_module_resource
from ...utils.utils import clean_text_linebreak
from ...exceptions import *
from ...loggers import logger


class AnyOCR(BaseOCRClient):
    def __init__(
        self,
        config: dict
    ):
        self.config = config
        self.image_batch_size = self.config.get("image_batch_size",1)
        self.ocr_batch_size = self.config.get("ocr_batch_size",1)
        ### 加载文档倾斜模型
        logger.debug(f"Load model: {self.__class__.__name__}.{DocOriClsModel.__name__}")
        self.doc_cls_config = self.config["doc_cls"]
        self.doc_cls_model = DocOriClsModel(config = self.doc_cls_config)
        
        ### 加载文档扭曲矫正模型
        logger.debug(f"Load model: {self.__class__.__name__}.{DocRectifierModel.__name__}")
        self.doc_rectifier_config = self.config["doc_rectifier"]
        self.doc_rectifier_model = DocRectifierModel(config = self.doc_rectifier_config)
        
        ### 加载文档布局分析模型
        logger.debug(f"Load model: {self.__class__.__name__}.{AnyDocLayoutV3.__name__}")
        self.layout_config = self.config['layout']
        self.layout_model = AnyDocLayoutV3(config = self.layout_config)
        
        ### 加载ocr模型
        self.ocr_config = self.config['vlm']
        self.ocr_model_type = self.ocr_config.get("model_type")
        if not self.ocr_model_type:
            raise AnyValueError("ocr_model_type is required")
        self.ocr_model_config = self.ocr_config.get(self.ocr_model_type)
        if not self.ocr_model_config:
            raise AnyValueError("ocr_model_config is required")
        self.ocr_model_class = self.ocr_model_config.pop("model_class",None)
        if not self.ocr_model_class:
            raise AnyValueError("ocr_model_class is required")
        self.ocr_model = import_module_resource(
                prefix_name = ".models",
                model_class = self.ocr_model_class,
                package = __package__
            )
        self.ocr_model = self.ocr_model(**self.ocr_model_config)
        logger.debug(f"Load model: {self.__class__.__name__}.{self.ocr_model.__class__.__name__}")

    def _build_payload(
        self,
        image_batch_size: int = None,
        ocr_batch_size: int = None,
        use_doc_cls: bool = None,
        use_doc_rectifier: bool = None,
        use_doc_layout: bool = None,
        doc_layout_image_min_size: int = None,
        use_image_resize: bool = False,
        max_new_tokens: int = 16384,
    ):
        if image_batch_size is None:
            image_batch_size = self.image_batch_size
        if ocr_batch_size is None:
            ocr_batch_size = self.ocr_batch_size
            
        if use_doc_cls is None:
            use_doc_cls = self.config.get('use_doc_cls', False)
            
        if use_doc_rectifier is None:
            use_doc_rectifier = self.config.get('use_doc_rectifier', False)
        
        if use_doc_layout is None:
            use_doc_layout = self.config.get("use_doc_layout", True)
            
        if doc_layout_image_min_size is None:
            doc_layout_image_min_size = self.config.get("doc_layout_image_min_size", 400)
            
        if use_image_resize is None:
            use_image_resize = self.config.get("use_image_resize", False)
        
        payload = {
            "image_batch_size": image_batch_size,
            "ocr_batch_size": ocr_batch_size,
            "use_doc_cls": use_doc_cls,
            "use_doc_rectifier": use_doc_rectifier,
            "use_doc_layout": use_doc_layout,
            "doc_layout_image_min_size": doc_layout_image_min_size,
            "use_image_resize": use_image_resize,
            "max_new_tokens": max_new_tokens
        }
        
        return payload
    
    def preprocess_images(
        self,
        images: List[Image.Image], 
        payload: dict,
    ):
        batch_range = range(0, len(images), payload["image_batch_size"])
        for batch_idx, i in enumerate(batch_range):
            batch_images = []
            for image in images[i: i + payload["image_batch_size"]]:
                if image.mode != "RGB":
                    image = image.convert("RGB")          
                batch_images.append(image)      
            if payload["use_doc_cls"]:
                cls_output = self.doc_cls_model.invoke(
                    images = batch_images,
                    batch_size = payload["image_batch_size"]
                )
                logger.debug(f"文档方向: [{cls_output}]")
                batch_images = [(x,y.label) for x,y in zip(batch_images, cls_output)]
                batch_images = self.doc_cls_model.invoke_rotate(
                    images = batch_images
                )
                batch_images = [cv2_to_pillow(x) for x in batch_images]
            if payload["use_doc_rectifier"]:
                try:
                    rectify_output = self.doc_rectifier_model.invoke(
                        images = batch_images,
                        batch_size = payload["image_batch_size"]
                    )
                    batch_images = [cv2_to_pillow(x.image) for x in rectify_output]
                    logger.debug(f"文档矫正: success")
                except:
                    traceback.print_exc()
            yield (batch_idx, batch_images)
        
    def build_layout_item(
        self,
        need_layout: List,
        not_layout: List,
        layout_outputs: List
    ):
        try:
            need_layout_res = []
            for (idx, image),layout_output in zip(need_layout, layout_outputs):
                need_layout_res.append(
                    (idx, image, layout_output)
                )
            batch_layout_outputs = not_layout + need_layout_res
            batch_layout_outputs = list(sorted(batch_layout_outputs, key = lambda x: x[0], reverse = False))
        except:
            traceback.print_exc()
            batch_layout_outputs = []
        
        finally:
            return batch_layout_outputs        
    
    def run_layout(
        self,
        batch_images: List[Image.Image],
        payload: dict
    ):
        try:
            need_layout = []
            not_layout = []
            for idx,image in enumerate(batch_images):
                width, height = image.size
                max_dimension = max(width, height) 
                if max_dimension <= payload["doc_layout_image_min_size"]:
                    logger.debug(f'文档布局分析自动跳过: {max_dimension} <= {payload["doc_layout_image_min_size"]}')
                    layout_output = self.layout_model.invoke(
                        images = [image],
                        batch_size = 1,
                        use_doc_layout = False
                    )
                    layout_output = layout_output[0] if layout_output else []
                    not_layout.append([idx, image, layout_output])
                else:
                    need_layout.append([idx, image])
            layout_outputs = self.layout_model.invoke(
                images = [x[-1] for x in need_layout],
                batch_size = payload["image_batch_size"],
                use_doc_layout = payload["use_doc_layout"]
            )
            batch_layout_outputs = self.build_layout_item(
                need_layout = need_layout,
                not_layout = not_layout,
                layout_outputs = layout_outputs
            )
            logger.debug(f"文档布局分析: success")
        
        except:
            traceback.print_exc()
            batch_layout_outputs = []
        
        finally:
            return batch_layout_outputs
        
    async def arun_layout(
        self,
        batch_images: List[Image.Image],
        payload: dict
    ):
        try:
            need_layout = []
            not_layout = []
            for idx,image in enumerate(batch_images):
                width, height = image.size
                max_dimension = max(width, height) 
                if max_dimension <= payload["doc_layout_image_min_size"]:
                    layout_output = await self.layout_model.ainvoke(
                        images = [image],
                        batch_size = 1,
                        use_doc_layout = False
                    )
                    layout_output = layout_output[0] if layout_output else []
                    not_layout.append([idx, image, layout_output])
                else:
                    need_layout.append([idx, image])
            layout_outputs = await self.layout_model.ainvoke(
                images = [x[-1] for x in need_layout],
                batch_size = payload["image_batch_size"],
                use_doc_layout = payload["use_doc_layout"]
            )
            batch_layout_outputs = self.build_layout_item(
                need_layout = need_layout,
                not_layout = not_layout,
                layout_outputs = layout_outputs
            )
            logger.debug(f"文档布局分析: success")
        
        except:
            traceback.print_exc()
            batch_layout_outputs = []
        
        finally:
            return batch_layout_outputs
    
    def build_ocr_item(
        self,
        need_ocr: List,
        ocr_res: List
    ):
        try:
            item = AnyOCROutput()
            ocr_outputs = []
            for (idx, crop_image, line), image_content in zip(need_ocr, ocr_res):
                ocr_outputs.append(
                    [idx, line, image_content]
                )
            image_outputs = list(sorted(ocr_outputs, key = lambda x: x[0], reverse = False))
            for idx,line,image_content in image_outputs:
                # print("image_content::: ", image_content)
                image_content = image_content.get("content", "")
                image_content = clean_text_linebreak(image_content)
                block = Element(
                    order_id = int(line['index']),
                    label = line['label'],
                    box = line['bbox'],
                    content = image_content
                )
                item.blocks.append(block)  
        except:
            traceback.print_exc()
            item = AnyOCROutput()
        
        finally:
            return item

    def run_ocr(
        self,
        batch_layout_outputs: List[Tuple[int, Image.Image, List[Dict]]],
        payload: dict,
        **kwargs
    ):
        try:
            output = []
            for image_idx, image, layout_output in batch_layout_outputs:
                need_ocr = []
                for idx,line in enumerate(layout_output):
                    box = tuple(line['bbox'])
                    crop_image = image.crop(box)
                    need_ocr.append(
                        [idx, crop_image, line]
                    )
                ocr_inputs = [(x[1], x[2]) for x in need_ocr]
                ocr_res = self.ocr_model.invoke(
                    images = ocr_inputs,
                    batch_size = payload["ocr_batch_size"],
                    use_image_resize = payload["use_image_resize"],
                    max_new_tokens = payload["max_new_tokens"],
                    **kwargs
                )
                item = self.build_ocr_item(
                    need_ocr = need_ocr,
                    ocr_res = ocr_res                    
                )
                output.append(item)          
            logger.debug(f"文档OCR: success")
        
        except:
            traceback.print_exc()
            output = []
        
        finally:
            return output        
        
    async def arun_ocr(
        self,
        batch_layout_outputs: List[Tuple[int, Image.Image, List[Dict]]],
        payload: dict,
        **kwargs
    ):
        try:
            output = []
            for image_idx, image, layout_output in batch_layout_outputs:
                need_ocr = []
                for idx,line in enumerate(layout_output):
                    box = tuple(line['bbox'])
                    crop_image = image.crop(box)
                    need_ocr.append(
                        [idx, crop_image, line]
                    )
                ocr_inputs = [(x[1], x[2]) for x in need_ocr]
                ocr_res = await self.ocr_model.ainvoke(
                    images = ocr_inputs,
                    batch_size = payload["ocr_batch_size"],
                    use_image_resize = payload["use_image_resize"],
                    max_new_tokens = payload["max_new_tokens"],
                    **kwargs
                )
                # print("ocr_res::: ", ocr_res)
                item = self.build_ocr_item(
                    need_ocr = need_ocr,
                    ocr_res = ocr_res                    
                )
                output.append(item)          
            logger.debug(f"文档OCR: success")
        
        except:
            traceback.print_exc()
            output = []
        
        finally:
            return output     

    def invoke(
        self, 
        images: List[Image.Image], 
        image_batch_size: int = None,
        ocr_batch_size: int = None,
        use_doc_cls: bool = None,
        use_doc_rectifier: bool = None,
        use_doc_layout: bool = None,
        doc_layout_image_min_size: int = None,
        use_image_resize: bool = False,
        max_new_tokens: int = 16384,
        **kwargs
    ) -> List[AnyOCROutput]:
        try:
            payload = self._build_payload(
                image_batch_size = image_batch_size,
                ocr_batch_size = ocr_batch_size,
                use_doc_cls = use_doc_cls,
                use_doc_rectifier = use_doc_rectifier,
                use_doc_layout = use_doc_layout,
                doc_layout_image_min_size = doc_layout_image_min_size,
                use_image_resize = use_image_resize,
                max_new_tokens = max_new_tokens
            )
            output = []
            batch_iterdata = self.preprocess_images(
                images = images,
                payload = payload
            )
            for (batch_idx, batch_images) in batch_iterdata:
                batch_num = len(batch_images)
                try:
                    batch_layout_outputs = self.run_layout(
                        batch_images = batch_images,
                        payload = payload,
                    )
                    batch_ocr_outputs = self.run_ocr(
                        batch_layout_outputs = batch_layout_outputs,
                        payload = payload,
                        **kwargs
                    )
                except:
                    traceback.print_exc()
                    batch_ocr_outputs = [""] * batch_num
                finally:
                    output.extend(batch_ocr_outputs)
            return output
        
        except:
            traceback.print_exc()
            output = []
        
        finally:
            return output             
        
    async def ainvoke(
        self, 
        images: List[Image.Image], 
        image_batch_size: int = None,
        ocr_batch_size: int = None,
        use_doc_cls: bool = None,
        use_doc_rectifier: bool = None,
        use_doc_layout: bool = None,
        doc_layout_image_min_size: int = None,
        use_image_resize: bool = False,
        max_new_tokens: int = 16384,
        **kwargs
    ) -> List[AnyOCROutput]:
        try:
            payload = self._build_payload(
                image_batch_size = image_batch_size,
                ocr_batch_size = ocr_batch_size,
                use_doc_cls = use_doc_cls,
                use_doc_rectifier = use_doc_rectifier,
                use_doc_layout = use_doc_layout,
                doc_layout_image_min_size = doc_layout_image_min_size,
                use_image_resize = use_image_resize,
                max_new_tokens = max_new_tokens
            )
            output = []
            batch_iterdata = self.preprocess_images(
                images = images,
                payload = payload
            )
            for (batch_idx, batch_images) in batch_iterdata:
                batch_num = len(batch_images)
                try:
                    batch_layout_outputs = await self.arun_layout(
                        batch_images = batch_images,
                        payload = payload,
                    )
                    batch_ocr_outputs = await self.arun_ocr(
                        batch_layout_outputs = batch_layout_outputs,
                        payload = payload,
                        **kwargs
                    )
                except:
                    traceback.print_exc()
                    batch_ocr_outputs = [""] * batch_num
                finally:
                    output.extend(batch_ocr_outputs)
            return output
        
        except:
            traceback.print_exc()
            output = []
        
        finally:
            return output             