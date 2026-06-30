import traceback
import torch
import asyncio
from typing import Any, Dict, Optional, Union, List
import numpy as np
from PIL import Image
from pathlib import Path
from transformers import (
    PPDocLayoutV3ForObjectDetection,
    PPDocLayoutV3ImageProcessor
)
from .base import BaseLayoutModel
from .utils import apply_layout_postprocess
from ....schemas import AnyDataModel,Field
from ....exceptions import *


class LayoutConfig(AnyDataModel):
    model_path: Optional[str] = 'resource/models/anydoclayout'
    threshold: float = 0.3
    threshold_by_class: Optional[Dict[Union[int, str], float]] = None
    batch_size: int = 1
    dtype: str = "auto"
    device_map: str = "auto"
    img_size: Optional[int] = None
    layout_nms: bool = True
    layout_unclip_ratio: Optional[Any] = [1.0, 1.0]
    layout_merge_bboxes_mode: Union[str, Dict[int, str]] = {
        0: 'large', 
        1: 'large', 
        2: 'large', 
        3: 'large', 
        4: 'large', 
        5: 'large', 
        6: 'large', 
        7: 'large', 
        8: 'large', 
        9: 'large', 
        10: 'large', 
        11: 'large', 
        12: 'large', 
        13: 'large', 
        14: 'large', 
        15: 'large', 
        16: 'large', 
        17: 'large', 
        18: 'small', 
        19: 'large', 
        20: 'large', 
        21: 'large', 
        22: 'large', 
        23: 'large', 
        24: 'large'
    }
    label_task_abandon: Optional[List[str]] = []
    # label_task_mapping: Optional[Dict[str, Any]] = {
    #         'text': [
    #                 'abstract', 
    #                 'algorithm', 
    #                 'content', 
    #                 'doc_title', 
    #                 'figure_title', 
    #                 'paragraph_title', 
    #                 'reference_content', 
    #                 'text', 
    #                 'vertical_text', 
    #                 'vision_footnote', 
    #                 'seal', 
    #                 'formula_number',
    #                 'header', 
    #                 'footer', 
    #                 'number', 
    #                 'footnote', 
    #                 'aside_text', 
    #                 'reference', 
    #                 'footer_image', 
    #                 'header_image'
    #             ], 
    #         'table': ['table'], 
    #         'formula': ['display_formula', 'inline_formula'], 
    #         'image': ['chart', 'image'],
    #         'skip': []
    #     }
    id2label: Optional[Dict[Any,Any]] = {
        0: 'abstract',            # 0 论文摘要
        1: 'algorithm',           # 1 算法
        2: 'aside_text',          # 2 页边注文本，通常位于页面边缘，提供补充信息或注释，与主内容相关但不直接包含在内
        3: 'chart',               # 3 图表，通常包含数据可视化元素，如柱状图、折线图、饼图等，用于展示数据关系和趋势
        4: 'content',             # 4 只在大的目录块中出现，其他地方未见
        5: 'display_formula',     # 5 独立展示的公式，通常占据整行或多行，具有较大字体和清晰的布局，以突出其重要性和可读性
        6: 'doc_title',           # 6 文章标题，一篇文章的主标题
        7: 'figure_title',        # 7 image/chart/table的caption
        8: 'footer',              # 8 页脚文本
        9: 'footer_image',        # 9 页脚图片
        10: 'footnote',           # 10 page footnote，通常位于页面底部，提供对正文中特定内容的补充说明、引用来源或其他相关信息
        11: 'formula_number',     # 11 公式编号，通常与display_formula配合使用，标识独立展示的公式在文档中的位置和顺序，便于引用和索引
        12: 'header',             # 12 页眉文本
        13: 'header_image',       # 13 页眉图片
        14: 'image',              # 14 图片
        15: 'inline_formula',     # 15 行内公式
        16: 'number',             # 16 页码
        17: 'paragraph_title',    # 17 段落标题，有别与文章标题
        18: 'reference',          # 18 参考文献，list外框
        19: 'reference_content',  # 19 参考文献内容，list item
        20: 'seal',               # 20 印章
        21: 'table',              # 21 表格
        22: 'text',               # 22 一般文本
        23: 'vertical_text',      # 23 竖排文本
        24: 'vision_footnote'     # 24 image/chart/table的footnote
    }
    

class AnyDocLayoutV3(BaseLayoutModel):
    """layout detector.

    Single instance, in-process batch inference. No multiprocessing workers.
    """

    def __init__(self, config: dict):
        """Initialize.

        Args:
            config: LayoutConfig instance.
        """
        self.config = LayoutConfig(**config)
        self.model_path = Path(self.config.model_path).expanduser().resolve()
        if not self.model_path.exists():
            raise AnyFileNotFoundError(f"Model path {self.model_path} does not exist.")
        
        ### 校验macos后端不支持float64
        if torch.backends.mps.is_available() and self.config.dtype in ["auto", "float64"]:
            print(f"Warning: MacOS backend does not support float64, auto fallback to float32 and cpu.")
            self.config.dtype = torch.float32
            self.config.device_map = "cpu"
        self.threshold = self.config.threshold
        self.layout_nms = self.config.layout_nms
        self.layout_unclip_ratio = self.config.layout_unclip_ratio
        self.layout_merge_bboxes_mode = self.config.layout_merge_bboxes_mode
        self.batch_size = self.config.batch_size
        self.label_task_abandon = self.config.label_task_abandon
        # self.label_task_mapping = self.config.label_task_mapping
        self.id2label = self.config.id2label
        self._model = None
        self._image_processor = None
        self._device = None
        self.start()

    def start(self):
        """Load model and processor once in the main process."""
        self._image_processor = PPDocLayoutV3ImageProcessor.from_pretrained(
            self.model_path
        )
        self._model = PPDocLayoutV3ForObjectDetection.from_pretrained(
            self.model_path,
            dtype = self.config.dtype,
            device_map = self.config.device_map
            # device_map = "cpu"
        )
        self._model.eval()
        self._device = self._model.device
        if self.id2label is None:
            self.id2label = self._model.config.id2label

    def stop(self):
        """Unload model and processor."""
        if self._model is not None:
            torch.cuda.empty_cache()
            self._model = None
        self._image_processor = None
        self._device = None
        
    def invoke(
        self,
        images: List[Image.Image],
        batch_size: int = 1,
        use_doc_layout: bool = True,
    ) -> List[List[Dict]]:
        """Batch-detect layout regions in-process.

        Args:
            images: List of PIL Images.
            save_visualization: Whether to also save visualization.
            visualization_output_dir: Where to save visualization outputs.
            global_start_idx: Start index for visualization filenames (layout_page{N}).

        Returns:
            List[List[Dict]]: Detection results per image.
        """
        if not use_doc_layout:
            all_results = []
            for image in images:
                width, height = image.size
                result = [
                    {
                        "index": 0,
                        "label": 'text',
                        "score": 0.0,
                        "bbox": [0,0,int(width),int(height)]
                    }                       
                ]
                all_results.append(result)
            return all_results
        
        if self._model is None:
            raise RuntimeError("Layout detector not started. Call start() first.")
  
        num_images = len(images)
        image_batch = []
        for image in images:
            image_width, image_height = image.size
            image_array = np.array(image.convert("RGB"))
            image_batch.append((image_array, image_width, image_height))

        pil_images = [Image.fromarray(img[0]) for img in image_batch]
        all_paddle_format_results = []

        if batch_size is None:
            batch_size = self.batch_size

        for chunk_start in range(0, num_images, batch_size):
            chunk_end = min(chunk_start + batch_size, num_images)
            chunk_pil = pil_images[chunk_start:chunk_end]

            inputs = self._image_processor(images=chunk_pil, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

            target_sizes = torch.tensor(
                [img.size[::-1] for img in chunk_pil], device=self._device
            )
            try:
                if hasattr(outputs, "pred_boxes") and outputs.pred_boxes is not None:
                    pred_boxes = outputs.pred_boxes
                    if hasattr(outputs, "out_masks") and outputs.out_masks is not None:
                        mask_h, mask_w = outputs.out_masks.shape[-2:]
                    else:
                        mask_h, mask_w = 200, 200
                    min_norm_w = 1.0 / mask_w
                    min_norm_h = 1.0 / mask_h
                    box_wh = pred_boxes[..., 2:4]
                    valid_mask = (box_wh[..., 0] > min_norm_w) & (
                        box_wh[..., 1] > min_norm_h
                    )
                    if hasattr(outputs, "logits") and outputs.logits is not None:
                        invalid_mask = ~valid_mask
                        if invalid_mask.any():
                            outputs.logits.masked_fill_(
                                invalid_mask.unsqueeze(-1), -100.0
                            )
            except Exception as e:
                traceback.print_exc()

            raw_results = self._image_processor.post_process_object_detection(
                outputs,
                threshold=self.threshold,
                target_sizes=target_sizes,
            )
            img_sizes = [img.size for img in chunk_pil]
            paddle_format_results = apply_layout_postprocess(
                raw_results=raw_results,
                id2label=self.id2label,
                img_sizes=img_sizes,
                layout_nms=self.layout_nms,
                layout_unclip_ratio=self.layout_unclip_ratio,
                layout_merge_bboxes_mode=self.layout_merge_bboxes_mode,
            )
            all_paddle_format_results.extend(paddle_format_results)

            if self._device.type.startswith("cuda") and chunk_end < num_images:
                del inputs, outputs, raw_results
                torch.cuda.empty_cache()

        all_results = []
        for img_idx, paddle_results in enumerate(all_paddle_format_results):
            image_width = image_batch[img_idx][1]
            image_height = image_batch[img_idx][2]
            results = []
            valid_index = 0
            for item in paddle_results:
                label = item["label"]
                score = item["score"]
                box = item["coordinate"]
                task_type = None
                if label in self.label_task_abandon:
                    task_type = "abandon"
                    continue
                # for task_item, labels in self.label_task_mapping.items():
                #     if isinstance(labels, list) and label in labels:
                #         task_type = task_item
                #         break
                # if task_type is None or task_type == "abandon":
                #     continue
                x1, y1, x2, y2 = box
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image_width, x2)
                y2 = min(image_height, y2)
                box = [x1, y1, x2, y2]
                box = [int(xx) for xx in box]
                results.append(
                    {
                        "index": valid_index,
                        "label": label.lower(),
                        "score": float(score),
                        "bbox": box
                    }
                )
                valid_index += 1
            all_results.append(results)

        return all_results

    async def ainvoke(
        self,
        images: List[Image.Image],
        batch_size: int = 1,
        use_doc_layout: bool = True,
    ) -> List[List[Dict]]:
        res = await asyncio.to_thread(
            self.invoke,
            images,
            batch_size,
            use_doc_layout,
        )
        return res