import copy
import traceback
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from omegaconf import DictConfig
from pydantic import BaseModel
from .calboxes import CalRecBoxes
from .clsmodel import TextClassifier, TextClsOutput
from .detmodel import TextDetector, TextDetOutput
from .recmodel import TextRecInput, TextRecognizer, TextRecOutput
from .loadImage import LoadImage
from .params import ParseParams
from .utils import (
    add_round_letterbox,
    get_padding_h,
    get_rotate_crop_image,
    resize_image_within_bounds,
)


class ToMarkdown:
    @classmethod
    def to(cls, boxes, txts) -> str:
        # def to(cls, result: AnyOCRBaseOutput) -> str:
        """
        根据 OCR 结果的坐标信息，将文本还原为近似原始排版的 Markdown。

        Args:
            result (AnyOCRBaseOutput): AnyOCR 的输出结果对象。

        Returns:
            str: 模拟原始排版的 Markdown 字符串。
        """
        if boxes is None or txts is None:
            return "没有检测到任何文本。"
        boxes = np.array(boxes)
        # 1. 将 box 和 text 绑定并排序
        #    主键：box 的顶部 y 坐标；次键：box 的左侧 x 坐标
        combined_data = sorted(
            zip(boxes, txts),
            key=lambda item: (
                cls.get_box_properties(item[0])["top"],
                cls.get_box_properties(item[0])["left"],
            ),
        )

        output_lines = []
        if not combined_data:
            return ""

        # 初始化当前行和前一个框的属性
        current_line_parts = [combined_data[0][1]]
        prev_props = cls.get_box_properties(combined_data[0][0])

        # 从第二个框开始遍历
        for box, text in combined_data[1:]:
            current_props = cls.get_box_properties(box)

            # 启发式规则来决定如何布局
            # 条件1：中心线距离是否足够近
            min_height = min(current_props["height"], prev_props["height"])
            centers_are_close = abs(
                current_props["center_y"] - prev_props["center_y"]
            ) < (min_height * 0.5)

            # 条件2：是否存在垂直方向的重叠
            # 计算重叠区域的顶部和底部
            overlap_top = max(prev_props["top"], current_props["top"])
            overlap_bottom = min(prev_props["bottom"], current_props["bottom"])
            has_vertical_overlap = overlap_bottom > overlap_top

            # 最终判断：满足任一条件即可
            is_same_line = centers_are_close or has_vertical_overlap

            if is_same_line:
                # 在同一行，用空格隔开
                current_line_parts.append("   ")  # 使用多个空格以产生明显间距
                current_line_parts.append(text)
            else:
                # 不在同一行，需要换行
                # 先将上一行组合成字符串并添加到输出列表
                output_lines.append("".join(current_line_parts))

                # 规则2：判断是否需要插入空行（新段落）
                # 如果垂直间距大于上一个框高度的某个比例（如70%），则认为是一个新段落
                vertical_gap = current_props["top"] - prev_props["bottom"]
                if vertical_gap > prev_props["height"] * 0.7:
                    output_lines.append("")  # 插入空行来创建段落

                # 开始一个新行
                current_line_parts = [text]

            # 更新前一个框的属性
            prev_props = current_props

        # 添加最后一行
        output_lines.append("".join(current_line_parts))
        return "\n".join(output_lines)

    @staticmethod
    def get_box_properties(box: np.ndarray) -> dict:
        """从坐标数组中计算框的几何属性"""
        # box shape is (4, 2) -> [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        ys = box[:, 1]
        xs = box[:, 0]

        top = np.min(ys)
        bottom = np.max(ys)
        left = np.min(xs)

        return {
            "top": top,
            "bottom": bottom,
            "left": left,
            "height": bottom - top,
            "center_y": top + (bottom - top) / 2,
        }


class AnyOCRBaseOutput(BaseModel):
    boxes: Optional[list] = None
    txts: Optional[tuple] = None
    scores: Optional[tuple] = None
    line_results: Optional[tuple] = None
    word_results: Optional[tuple] = None
    elapse_times: Optional[dict] = None
    lang_type: Optional[str] = None
    
    def __len__(self):
        if self.txts is None:
            return 0
        return len(self.txts)

    def to_markdown(self) -> str:
        return ToMarkdown.to(self.boxes, self.txts)    
    

class AnyOCRError(Exception):
    pass


class AnyOCRbase:
    def __init__(self, config):
        cfg = self.load_config(config)
        self.initialize(cfg)

    def load_config(self, config) -> DictConfig:
        cfg = ParseParams.load(config)
        return cfg

    def initialize(self, cfg: DictConfig):
        self.text_score = cfg.Global.text_score
        self.min_height = cfg.Global.min_height
        self.width_height_ratio = cfg.Global.width_height_ratio

        self.use_det = cfg.Global.use_det
        cfg.Det.engine_cfg = cfg.EngineConfig[cfg.Det.engine_type.value]
        self.text_det = TextDetector(cfg.Det)

        self.use_cls = cfg.Global.use_cls
        cfg.Cls.engine_cfg = cfg.EngineConfig[cfg.Cls.engine_type.value]
        self.text_cls = TextClassifier(cfg.Cls)

        self.use_rec = cfg.Global.use_rec
        cfg.Rec.engine_cfg = cfg.EngineConfig[cfg.Rec.engine_type.value]
        self.text_rec = TextRecognizer(cfg.Rec)

        self.load_img = LoadImage()
        self.max_side_len = cfg.Global.max_side_len
        self.min_side_len = cfg.Global.min_side_len

        self.cal_rec_boxes = CalRecBoxes()

        self.return_word_box = cfg.Global.return_word_box
        self.return_single_char_box = cfg.Global.return_single_char_box

        self.cfg = cfg

    def invoke(
        self,
        img_content: Union[str, np.ndarray, Image.Image, bytes, Path],
        use_det: Optional[bool] = None,
        use_cls: Optional[bool] = None,
        use_rec: Optional[bool] = None,
        cls_line: Optional[bool] = True,
        return_word_box: bool = True,
        return_single_char_box: bool = True,
        text_score: float = 0.5,
        box_thresh: float = 0.5,
        unclip_ratio: float = 1.6,
        **kwargs
    ) -> Union[TextDetOutput, TextClsOutput, TextRecOutput, AnyOCRBaseOutput]:
        self.update_params(
            use_det,
            use_cls,
            use_rec,
            return_word_box,
            return_single_char_box,
            text_score,
            box_thresh,
            unclip_ratio,
        )

        ori_img = self.load_img(img_content)
        img, op_record = self.preprocess_img(ori_img)

        det_res, cls_res, rec_res = TextDetOutput(), TextClsOutput(), TextRecOutput()
        if self.use_det:
            try:
                img, det_res = self.get_det_res(img, op_record)
            except Exception as e:
                traceback.print_exc()
                return AnyOCRBaseOutput()

        if self.use_cls:
            try:
                img, cls_res = self.get_cls_res(img)
            except Exception as e:
                traceback.print_exc()
                return AnyOCRBaseOutput()

        if self.use_rec:
            rec_res = self.get_rec_res(img)

        return self.finalize_results(ori_img, det_res, cls_res, rec_res, img, op_record)

    def finalize_results(
        self,
        ori_img: np.ndarray,
        det_res: TextDetOutput,
        cls_res: TextClsOutput,
        rec_res: TextRecOutput,
        img: List[np.ndarray],
        op_record: Dict[str, Any],
    ) -> Union[TextDetOutput, TextClsOutput, TextRecOutput, AnyOCRBaseOutput]:
        raw_h, raw_w = ori_img.shape[:2]

        if (
            self.return_word_box
            and det_res.boxes is not None
            and all(v for v in rec_res.word_results)
        ):
            rec_res.word_results = self.calc_word_boxes(
                img, det_res, rec_res, op_record, raw_h, raw_w
            )

        if det_res.boxes is not None:
            det_res.boxes = self._get_origin_points(
                det_res.boxes, op_record, raw_h, raw_w
            )
        return self.get_final_res(ori_img, det_res, cls_res, rec_res)

    def calc_word_boxes(
        self,
        img: List[np.ndarray],
        det_res: TextDetOutput,
        rec_res: TextRecOutput,
        op_record: Dict[str, Any],
        raw_h: int,
        raw_w: int,
    ) -> Any:
        rec_res = self.cal_rec_boxes(
            img, det_res.boxes, rec_res, self.return_single_char_box
        )

        origin_words = []
        for word_line in rec_res.word_results:
            origin_words_item = []
            for txt, score, bbox in word_line:
                if bbox is None:
                    continue

                origin_words_points = self._get_origin_points(
                    [bbox], op_record, raw_h, raw_w
                )
                origin_words_points = origin_words_points.astype(np.int32).tolist()[0]
                origin_words_item.append((txt, score, origin_words_points))

            if origin_words_item:
                origin_words.append(tuple(origin_words_item))
        return tuple(origin_words)

    def update_params(
        self,
        use_det,
        use_cls,
        use_rec,
        return_word_box,
        return_single_char_box,
        text_score,
        box_thresh,
        unclip_ratio,
    ):
        self.use_det = self.use_det if use_det is None else use_det
        self.use_cls = self.use_cls if use_cls is None else use_cls
        self.use_rec = self.use_rec if use_rec is None else use_rec

        self.return_word_box = return_word_box
        self.return_single_char_box = return_single_char_box
        self.text_score = text_score
        self.text_det.postprocess_op.box_thresh = box_thresh
        self.text_det.postprocess_op.unclip_ratio = unclip_ratio

    def preprocess_img(self, ori_img: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        op_record = {}
        img, ratio_h, ratio_w = resize_image_within_bounds(
            ori_img, self.min_side_len, self.max_side_len
        )
        op_record["preprocess"] = {"ratio_h": ratio_h, "ratio_w": ratio_w}
        return img, op_record

    def get_det_res(
        self, img: np.ndarray, op_record: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], TextDetOutput]:
        img, op_record = self._add_letterbox(img, op_record)
        det_res = self.text_det(img)
        if det_res.boxes is None:
            raise AnyOCRError("The text detection result is empty")

        img_list = self.get_crop_img_list(img, det_res)
        return img_list, det_res

    def get_crop_img_list(
        self, img: np.ndarray, det_res: TextDetOutput
    ) -> List[np.ndarray]:
        img_crop_list = []
        for box in det_res.boxes:
            tmp_box = copy.deepcopy(box)
            img_crop = get_rotate_crop_image(img, tmp_box)
            img_crop_list.append(img_crop)
        return img_crop_list

    def get_cls_res(
        self, img: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], TextClsOutput]:
        cls_res = self.text_cls(img)
        if cls_res.img_list is None:
            raise AnyOCRError("The text classifier is empty")
        return cls_res.img_list, cls_res

    def get_rec_res(self, img: List[np.ndarray]) -> TextRecOutput:
        rec_input = TextRecInput(img=img, return_word_box=self.return_word_box)
        return self.text_rec(rec_input)

    def get_final_res(
        self,
        ori_img: np.ndarray,
        det_res: TextDetOutput,
        cls_res: TextClsOutput,
        rec_res: TextRecOutput,
    ) -> Union[TextDetOutput, TextClsOutput, TextRecOutput, AnyOCRBaseOutput]:
        dt_boxes = det_res.boxes
        txt_res = rec_res.txts

        if dt_boxes is None and txt_res is None and cls_res.cls_res is not None:
            return cls_res

        if dt_boxes is None and txt_res is None:
            return AnyOCRBaseOutput()

        if dt_boxes is None and txt_res is not None:
            return rec_res

        if dt_boxes is not None and txt_res is None:
            return det_res
        
        ocr_res = AnyOCRBaseOutput(
            boxes=det_res.boxes.tolist(),
            txts=rec_res.txts,
            scores=rec_res.scores,
            word_results=rec_res.word_results,
            elapse_times={
                "det": det_res.elapse,
                "cls": cls_res.elapse,
                "rec": rec_res.elapse,
                "total": sum([det_res.elapse, cls_res.elapse, rec_res.elapse])
            },
            lang_type=self.cfg.Rec.lang_type,
        )
        ocr_res.line_results = self.cls_line_wordbox(ocr_res)
        ocr_res = self.filter_by_text_score(ocr_res)
        if len(ocr_res) <= 0:
            return AnyOCRBaseOutput()

        return ocr_res

    def filter_by_text_score(self, ocr_res: AnyOCRBaseOutput) -> AnyOCRBaseOutput:
        filter_boxes, filter_txts, filter_scores = [], [], []
        for box, txt, score in zip(ocr_res.boxes, ocr_res.txts, ocr_res.scores):
            if float(score) >= self.text_score:
                filter_boxes.append(box)
                filter_txts.append(txt)
                filter_scores.append(score)

        ocr_res.boxes = filter_boxes
        ocr_res.txts = tuple(filter_txts)
        ocr_res.scores = tuple(filter_scores)
        return ocr_res

    def _add_letterbox(
        self, img: np.ndarray, op_record: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        h, w = img.shape[:2]

        if self.width_height_ratio == -1:
            use_limit_ratio = False
        else:
            use_limit_ratio = w / h > self.width_height_ratio

        if h <= self.min_height or use_limit_ratio:
            padding_h = get_padding_h(h, w, self.width_height_ratio, self.min_height)
            block_img = add_round_letterbox(img, (padding_h, padding_h, 0, 0))
            op_record["padding_1"] = {"top": padding_h, "left": 0}
            return block_img, op_record

        op_record["padding_1"] = {"top": 0, "left": 0}
        return img, op_record

    def _get_origin_points(
        self,
        dt_boxes: List[np.ndarray],
        op_record: Dict[str, Any],
        raw_h: int,
        raw_w: int,
    ) -> np.ndarray:
        dt_boxes_array = np.array(dt_boxes).astype(np.float32)
        for op in reversed(list(op_record.keys())):
            v = op_record[op]
            if "padding" in op:
                top, left = v.get("top"), v.get("left")
                dt_boxes_array[:, :, 0] -= left
                dt_boxes_array[:, :, 1] -= top
            elif "preprocess" in op:
                ratio_h = v.get("ratio_h")
                ratio_w = v.get("ratio_w")
                dt_boxes_array[:, :, 0] *= ratio_w
                dt_boxes_array[:, :, 1] *= ratio_h

        dt_boxes_array = np.where(dt_boxes_array < 0, 0, dt_boxes_array)
        dt_boxes_array[..., 0] = np.where(
            dt_boxes_array[..., 0] > raw_w, raw_w, dt_boxes_array[..., 0]
        )
        dt_boxes_array[..., 1] = np.where(
            dt_boxes_array[..., 1] > raw_h, raw_h, dt_boxes_array[..., 1]
        )
        return dt_boxes_array
    
    def cls_line_wordbox(self, ocr_res):
        res = {}
        for idx,(txt,box) in enumerate(zip(ocr_res.txts,ocr_res.boxes)):
            item = [txt,box]
            if idx == 0:
                res["1"] = [item]
            else:
                x1,y1 = box[0]
                x2,y2 = box[2]
                reference_y = (y2 + y1) / 2
                flag = True
                for k in list(res.keys()):
                    start_box = res[k][0][-1]
                    start_y1 = start_box[0][-1]
                    start_y2 = start_box[2][-1]
                    start_reference_y = (start_y2 + start_y1) / 2
                    threshold = start_y2 - start_reference_y
                    if abs(start_reference_y - reference_y) < threshold:
                        res[k].append(item)
                        flag = False
                        break
                if flag:
                    item_key = f'{len(list(res.keys())) + 1}'
                    res[item_key] = [item]
        outputs = []
        for k in res.keys():
            line = [x[0] for x in res[k]]
            outputs.append(line)
        outputs = tuple(outputs)
        return outputs

