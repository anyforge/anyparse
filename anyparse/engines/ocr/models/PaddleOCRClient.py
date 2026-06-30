import re
import copy
import cv2
import torch
import asyncio
from PIL import Image
from pathlib import Path
from typing import List, Tuple
import numpy as np
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    AutoModelForTextRecognition
)
from .base import BaseOCRModel
from ....exceptions import *


class SortQuadBoxes(object):
    def __init__(self, **kwargs):
        """Initializes the class."""
        self.kwargs = kwargs

    def invoke(self, dt_polys: List[np.ndarray]) -> np.ndarray:
        """
        Sort quad boxes in order from top to bottom, left to right
        args:
            dt_polys(ndarray):detected quad boxes with shape [4, 2]
        return:
            sorted boxes(ndarray) with shape [4, 2]
        """
        dt_boxes = np.array(dt_polys)
        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)

        for i in range(num_boxes - 1):
            for j in range(i, -1, -1):
                if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and (
                    _boxes[j + 1][0][0] < _boxes[j][0][0]
                ):
                    tmp = _boxes[j]
                    _boxes[j] = _boxes[j + 1]
                    _boxes[j + 1] = tmp
                else:
                    break
        return _boxes
    
    def invoke_by_indices(self, dt_polys: List[np.ndarray]) -> np.ndarray:
        dt_boxes = np.array(dt_polys)
        num_boxes = dt_boxes.shape[0]
        
        # 【关键改动】：不再直接对 dt_boxes 排序，而是对索引进行排序
        indices = list(range(num_boxes))
        sorted_indices = sorted(indices, key=lambda i: (dt_boxes[i][0][1], dt_boxes[i][0][0]))
        
        _indices = list(sorted_indices)

        for i in range(num_boxes - 1):
            for j in range(i, -1, -1):
                # 【关键改动】：通过索引 i1, i2 去获取原始坐标进行比较
                y1, x1 = dt_boxes[_indices[j + 1]][0][1], dt_boxes[_indices[j + 1]][0][0]
                y2, x2 = dt_boxes[_indices[j]][0][1], dt_boxes[_indices[j]][0][0]
                
                if abs(y1 - y2) < 10 and x1 < x2:
                    tmp = _indices[j]
                    _indices[j] = _indices[j + 1]
                    _indices[j + 1] = tmp
                else:
                    break
                    
        return _indices  # 返回排好序的【索引列表】


class CropQuadBoxes(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        
    def get_rotate_crop_image(self, img: np.ndarray, points: list) -> np.ndarray:
        """
        Crop and rotate the input image based on the given four points to form a perspective-transformed image.

        Args:
            img (np.ndarray): The input image array.
            points (list): A list of four 2D points defining the crop region in the image.

        Returns:
            np.ndarray: The transformed image array.
        """
        assert len(points) == 4, "shape of points must be 4*2"
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3]),
            )
        )
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2]),
            )
        )
        pts_std = np.float32(
            [
                [0, 0],
                [img_crop_width, 0],
                [img_crop_width, img_crop_height],
                [0, img_crop_height],
            ]
        )
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M,
            (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC,
        )
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img
    
    def get_minarea_rect_crop(self, img: np.ndarray, points: np.ndarray) -> np.ndarray:
        """
        Get the minimum area rectangle crop from the given image and points.

        Args:
            img (np.ndarray): The input image.
            points (np.ndarray): A list of points defining the shape to be cropped.

        Returns:
            np.ndarray: The cropped image with the minimum area rectangle.
        """
        bounding_box = cv2.minAreaRect(np.array(points).astype(np.int32))
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_a, index_b, index_c, index_d = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_a = 0
            index_d = 1
        else:
            index_a = 1
            index_d = 0
        if points[3][1] > points[2][1]:
            index_b = 2
            index_c = 3
        else:
            index_b = 3
            index_c = 2

        box = [points[index_a], points[index_b], points[index_c], points[index_d]]
        crop_img = self.get_rotate_crop_image(img, np.array(box))
        return crop_img
        
    def invoke(self, img: np.ndarray, quad_points: List[list]) -> List[dict]:
        """
        Call method to crop images based on detection boxes.

        Args:
            img (nd.ndarray): The input image.
            quad_points (list[list]): List of detection points.

        Returns:
            list[dict]: A list of dictionaries containing cropped images and their sizes.
        """
        dt_boxes = np.array(quad_points)
        output_list = []
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_minarea_rect_crop(img, tmp_box)
            output_list.append(img_crop)

        return output_list


class CTCLabelDecode(object):
    def __init__(
        self,
        character_list = [],
        use_space_char: bool = True,
        add_special_char: bool = True,
        space_char: str = " ",
        special_char: str = "blank",
    ):
        self.use_space_char = use_space_char
        self.add_special_char = add_special_char
        self.space_char = space_char
        self.special_char = special_char
        self.reverse = False
        if not character_list:
            character_list = list("0123456789abcdefghijklmnopqrstuvwxyz")
        character_list = list(character_list)
        
        ## add space char
        if self.use_space_char and character_list[-1] != self.space_char:
            character_list.append(self.space_char)        
        
        ## add blank char
        if self.add_special_char and character_list[0] != self.special_char:
            character_list = [self.special_char] + character_list
        self.dict = {}
        for i, char in enumerate(character_list):
            self.dict[char] = i
        self.character = character_list        
        
    def get_ignored_tokens(self):
        """get_ignored_tokens"""
        return [0]  # for ctc blank

    def pred_reverse(self, pred):
        """pred_reverse"""
        pred_re = []
        c_current = ""
        for c in pred:
            if not bool(re.search("[a-zA-Z0-9 :*./%+-]", c)):
                if c_current != "":
                    pred_re.append(c_current)
                pred_re.append(c)
                c_current = ""
            else:
                c_current += c
        if c_current != "":
            pred_re.append(c_current)

        return "".join(pred_re[::-1])

    def decode(
        self,
        text_index,
        text_prob=None,
        is_remove_duplicate=False,
    ):
        """convert text-index into text-label."""
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            char_list = [
                self.character[text_id] for text_id in text_index[batch_idx][selection]
            ]
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection].tolist()
            else:
                conf_list = [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]

            text = "".join(char_list)

            if self.reverse:  # for arabic rec
                text = self.pred_reverse(text)

            result_list.append(
                (
                    text, 
                    conf_list,
                )
            )
        return result_list
    
    def invoke(
        self,
        preds
    ):
        preds = np.array(preds)
        preds_idx = preds.argmax(axis=-1)
        preds_prob = preds.max(axis=-1)
        text = self.decode(
            preds_idx,
            preds_prob,
            is_remove_duplicate=True
        )
        texts = []
        scores = []
        for t in text:
            texts.append(t[0] if len(t) <= 2 else (t[0], t[2]))
            scores.append(t[1])
        return texts, scores   


class PPOCRV6(BaseOCRModel):
    def __init__(
        self,
        model_path: str = "resource/models/paddleocrv6-small",
        # det_model_path: str = "resource/models/paddleocrv6-small/det",
        # rec_model_path: str = "resource/models/paddleocrv6-small/rec",
        batch_size: int = 8,
        dtype: str = "auto",
        device_map: str = "auto",
        rec_img_shape: list = [3, 48, 320],
        **kwargs
    ):
        det_model_path = Path(model_path) / "det"
        rec_model_path = Path(model_path) / "rec"
        self.det_model_path = Path(det_model_path).expanduser().resolve()
        self.rec_model_path = Path(rec_model_path).expanduser().resolve()
        if not self.det_model_path.exists():
            raise AnyFileNotFoundError(
                f"{self.det_model_path} not found!"
            )
            
        if not self.rec_model_path.exists():
            raise AnyFileNotFoundError(
                f"{self.rec_model_path} not found!"
            )

        self.batch_size = batch_size
        self.dtype = dtype
        self.device_map = device_map
        self.ocr_det_model = AutoModelForObjectDetection.from_pretrained(
            self.det_model_path, 
            dtype = self.dtype,
            device_map = self.device_map,
        ).eval()
        self.ocr_det_image_processor = AutoImageProcessor.from_pretrained(
            self.det_model_path
        )
        
        self.ocr_rec_model = AutoModelForTextRecognition.from_pretrained(
            self.rec_model_path, 
            torch_dtype = self.dtype,
            device_map = self.device_map,
        ).eval()
        self.ocr_rec_image_processor = AutoImageProcessor.from_pretrained(
            self.rec_model_path
        )
        self.character_list = self.ocr_rec_image_processor.character_list
        self.rec_img_shape = rec_img_shape or [
            3,
            self.ocr_rec_image_processor.size.height,
            self.ocr_rec_image_processor.size.width
        ]
        self.sort_utils = SortQuadBoxes()
        self.crop_utils = CropQuadBoxes() 
        self.ctc_decoder = CTCLabelDecode(
            character_list = self.character_list
        )   
        self.det_threshold = kwargs.get("det_threshold",0.3)
        self.det_box_threshold = kwargs.get("det_box_threshold",0.6)
        self.det_max_candidates = kwargs.get("det_max_candidates",4000)
        self.det_min_size = kwargs.get("det_min_size",3)
        self.det_unclip_ratio = kwargs.get("det_unclip_ratio",2.0)
        self.ocr_line_y_thresh = kwargs.get("ocr_line_y_thresh",15)
        
    def format_ocr_results(self, ocr_results, y_thresh=15):
        """
        将 OCR 结果按行分组并格式化为多行文本。
        
        Args:
            ocr_results (list): OCR 返回的结果列表，每个元素包含 'box' 和 'text'。
            y_thresh (int): Y 轴阈值，Y 坐标差距小于此值的会被视为同一行。
            
        Returns:
            str: 格式化后的多行文本字符串。
        """
        if not ocr_results:
            return ""

        # 1. 提取 Y 轴中心点并按 Y 排序，方便后续分组
        # box 格式为 [x1, y1, x2, y2]，我们取 (y1 + y2) / 2 作为中心 Y 坐标
        items = []
        for item in ocr_results:
            box = item['box']
            center_y = (box[1] + box[3]) / 2.0
            center_x = (box[0] + box[2]) / 2.0
            items.append((center_y, center_x, item['text']))
        
        # 先按 Y 坐标排序，再按 X 坐标排序
        items.sort(key=lambda x: (x[0], x[1]))

        # 2. 按照 Y 轴阈值进行行聚类
        lines = []
        current_line = [items[0]]
        current_y = items[0][0]

        for i in range(1, len(items)):
            cy, cx, text = items[i]
            # 如果当前元素的 Y 坐标与当前行的基准 Y 差距小于阈值，则归入当前行
            if abs(cy - current_y) < y_thresh:
                current_line.append((cy, cx, text))
            else:
                # 否则，开启新的一行
                lines.append(current_line)
                current_line = [(cy, cx, text)]
                current_y = cy
                
        # 别忘了把最后一行加入结果
        if current_line:
            lines.append(current_line)

        # 3. 格式化输出：同行用空格 join，不同行用换行符 \n 连接
        formatted_lines = []
        for line in lines:
            # 因为前面已经按 X 排过序了，这里直接提取文本并用空格拼接
            line_text = " ".join([item[2] for item in line])
            formatted_lines.append(line_text)

        return "\n".join(formatted_lines)
        
    def invoke(
        self, 
        images: List[Image.Image] | List[Tuple[Image.Image, dict]],
        rec_batch_size: int = 6,
        det_threshold: float = 0.3,
        det_box_threshold: float = 0.6,
        det_max_candidates: int = 4000,
        det_min_size: int = 3,
        det_unclip_ratio: float = 2.0,
        **kwargs
    ) -> List[dict]:
        if rec_batch_size is None:
            rec_batch_size = self.batch_size
        if det_threshold is None:
            det_threshold = self.det_threshold
        if det_box_threshold is None:
            det_box_threshold = self.det_box_threshold
        if det_max_candidates is None:
            det_max_candidates = self.det_max_candidates
        if det_min_size is None:
            det_min_size = self.det_min_size
        if det_unclip_ratio is None:
            det_unclip_ratio = self.det_unclip_ratio
        output = []
        for image_idx, item in enumerate(images):
            if isinstance(item, Image.Image):
                image = item
            elif isinstance(item, tuple) and len(item) == 2:
                    image, task_map = item
            else:
                continue
            if image.mode != "RGB":
                image = image.convert("RGB")
            det_inputs = self.ocr_det_image_processor(
                images = image, 
                return_tensors="pt"
            ).to(self.ocr_det_model.device)
            with torch.no_grad():
                det_outputs = self.ocr_det_model(**det_inputs)
            det_results = self.ocr_det_image_processor.post_process_object_detection(
                det_outputs, 
                target_sizes=det_inputs["target_sizes"],
                threshold = det_threshold,
                box_threshold = det_box_threshold,
                max_candidates = det_max_candidates,
                min_size = det_min_size,
                unclip_ratio = det_unclip_ratio            
            )    
            image_np = np.array(image) 
            ocr_det_output = []
            text_idx = 0
            for det_item in det_results:
                boxes = det_item['boxes'].detach().cpu().numpy()
                scores = det_item['scores'].detach().cpu().numpy()      
                quad_points = boxes.tolist()
                cropped_images = self.crop_utils.invoke(
                    image_np, quad_points
                )
                ocr_det_res = []
                for box,score,cropimg in zip(boxes,scores,cropped_images):
                    x_min, y_min = box.min(axis=0).tolist()
                    x_max, y_max = box.max(axis=0).tolist()
                    bbox = [x_min, y_min, x_max, y_max]
                    # poly = [tuple(bb) for bb in box.tolist()] 
                    poly = box 
                    score = score.item()
                    ocr_det_res.append({
                        "image_idx": image_idx,
                        "bbox": bbox,
                        "poly": poly,
                        "det_score": score,
                        "crop_image": cropimg,
                        "text_idx": text_idx
                    })
                    text_idx += 1
                # 1. 提取当前图片中所有的 poly
                current_polys = [item['poly'] for item in ocr_det_res]                
                # 2. 传入坐标，获取排好序的【索引列表】
                sorted_indices = self.sort_utils.invoke_by_indices(current_polys)
                # 3. 按照索引顺序，安全地提取字典，生成最终的有序列表
                sorted_ocr_det_res = [ocr_det_res[idx] for idx in sorted_indices]
                # 4. extend 到总输出列表中
                ocr_det_output.extend(sorted_ocr_det_res)
            
            one_output = []
            for j in range(0, len(ocr_det_output), rec_batch_size):
                batch_data = ocr_det_output[j:j+rec_batch_size]  
                batch_crop_images = [item['crop_image'].copy() for item in batch_data]
                batch_rec_inputs = self.ocr_rec_image_processor(
                    images = batch_crop_images, 
                    return_tensors="pt"
                ).to(self.ocr_rec_model.device)
                ### infer
                with torch.no_grad():
                    rec_output = self.ocr_rec_model(**batch_rec_inputs)
                    rec_logits = rec_output.last_hidden_state
                    preds = rec_logits.detach().cpu().numpy()
                texts, scores = self.ctc_decoder.invoke(
                    preds = preds
                )       
                for batch_idx, batch_item in enumerate(batch_data):
                    item_text = texts[batch_idx]
                    item_score = scores[batch_idx]
                    rec_score = np.mean(item_score).tolist()  
                    batch_item["poly"] = [tuple(bb) for bb in batch_item["poly"].tolist()] 
                    line = {
                        "image_idx": batch_item["image_idx"],
                        "text_idx": batch_item["text_idx"],
                        "box": batch_item["bbox"],
                        "poly": batch_item["poly"],
                        "det_score": batch_item["det_score"],
                        "text": item_text,
                        "rec_score": rec_score
                    }
                    one_output.append(line)
        
            format_output = []
            for idx,item in enumerate(one_output):
                item['text_idx'] = idx
                format_output.append(item)
            format_content = self.format_ocr_results(
                ocr_results = format_output, 
                y_thresh = kwargs.get("ocr_line_y_thresh", self.ocr_line_y_thresh)
            )
            output.append(
                {
                    "content": format_content,
                    "blocks": format_output
                }
            )
            output.append(format_output)    
            
        return output                

    async def ainvoke(
        self, 
        images: List[Image.Image] | List[Tuple[Image.Image, dict]],
        rec_batch_size: int = 8,
        det_threshold: float = 0.3,
        det_box_threshold: float = 0.6,
        det_max_candidates: int = 1000,
        det_min_size: int = 3,
        det_unclip_ratio: float = 1.5,
        **kwargs
    ) -> List[dict]:
        res = await asyncio.to_thread(
            self.invoke,
            images = images,
            rec_batch_size = rec_batch_size,
            det_threshold = det_threshold,
            det_box_threshold = det_box_threshold,
            det_max_candidates = det_max_candidates,
            det_min_size = det_min_size,
            det_unclip_ratio = det_unclip_ratio,
            **kwargs
        )
        return res


