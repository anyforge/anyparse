import cv2
import copy
import math
import time
import dataclasses
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from .engine import get_engine


@dataclasses.dataclass
class TextClsOutput:
    img_list: Optional[List[np.ndarray]] = None
    cls_res: Optional[List[Tuple[str, float]]] = None
    elapse: Optional[float] = None

    def __len__(self):
        if self.img_list is None:
            return 0
        return len(self.img_list)


class ClsPostProcess:
    def __init__(self, label_list: List[str]):
        self.label_list = label_list

    def __call__(self, preds: np.ndarray) -> List[Tuple[str, float]]:
        pred_idxs = preds.argmax(axis=1)
        decode_out = [
            (self.label_list[int(idx)], preds[i, int(idx)])
            for i, idx in enumerate(pred_idxs)
        ]
        return decode_out


class TextClassifier:
    def __init__(self, cfg: Dict[str, Any]):
        self.cls_image_shape = cfg["cls_image_shape"]
        self.cls_batch_num = cfg["cls_batch_num"]
        self.cls_thresh = cfg["cls_thresh"]
        self.postprocess_op = ClsPostProcess(cfg["label_list"])

        self.session = get_engine(cfg.engine_type)(cfg)

    def __call__(self, img_list: Union[np.ndarray, List[np.ndarray]]) -> TextClsOutput:
        if isinstance(img_list, np.ndarray):
            img_list = [img_list]

        img_list = copy.deepcopy(img_list)

        # Calculate the aspect ratio of all text bars
        width_list = [img.shape[1] / float(img.shape[0]) for img in img_list]

        # Sorting can speed up the cls process
        indices = np.argsort(np.array(width_list))

        img_num = len(img_list)
        cls_res = [("", 0.0)] * img_num
        batch_num = self.cls_batch_num
        elapse = 0
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)

            norm_img_batch = []
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]])
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch).astype(np.float32)

            starttime = time.time()
            prob_out = self.session(norm_img_batch)
            cls_result = self.postprocess_op(prob_out)
            elapse += time.time() - starttime

            for rno, (label, score) in enumerate(cls_result):
                cls_res[indices[beg_img_no + rno]] = (label, score)
                if "180" in label and score > self.cls_thresh:
                    img_list[indices[beg_img_no + rno]] = cv2.rotate(
                        img_list[indices[beg_img_no + rno]], 1
                    )
        return TextClsOutput(img_list=img_list, cls_res=cls_res, elapse=elapse)

    def resize_norm_img(self, img: np.ndarray) -> np.ndarray:
        img_c, img_h, img_w = self.cls_image_shape
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(img_h * ratio) > img_w:
            resized_w = img_w
        else:
            resized_w = int(math.ceil(img_h * ratio))

        resized_image = cv2.resize(img, (resized_w, img_h))
        resized_image = resized_image.astype("float32")
        if img_c == 1:
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
        else:
            resized_image = resized_image.transpose((2, 0, 1)) / 255

        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((img_c, img_h, img_w), dtype=np.float32)
        padding_im[:, :, :resized_w] = resized_image
        return padding_im
