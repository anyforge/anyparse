import cv2
import hashlib
import dataclasses
import numpy as np
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse
from dataclasses import dataclass, field
from typing import List, Tuple, Union


def quads_to_rect_bbox(bbox: np.ndarray) -> Tuple[float, float, float, float]:
    if bbox.ndim != 3:
        raise ValueError("bbox shape must be 3")

    if bbox.shape[1] != 4 and bbox.shape[2] != 2:
        raise ValueError("bbox shape must be (N, 4, 2)")

    all_x, all_y = (bbox[:, :, 0].flatten(), bbox[:, :, 1].flatten())
    x_min, y_min = np.min(all_x), np.min(all_y)
    x_max, y_max = np.max(all_x), np.max(all_y)
    return float(x_min), float(y_min), float(x_max), float(y_max)


def has_chinese_char(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def get_file_sha256(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    with open(file_path, "rb") as file:
        sha_signature = hashlib.sha256()
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            sha_signature.update(chunk)

    return sha_signature.hexdigest()


def is_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception as e:
        return False


class WordType(Enum):
    CN = "cn"
    EN = "en"
    NUM = "num"
    EN_NUM = "en&num"


@dataclass
class WordInfo:
    words: List[List[str]] = field(default_factory=list)
    word_cols: List[List[int]] = field(default_factory=list)
    word_types: List[WordType] = field(default_factory=list)
    line_txt_len: float = 0.0
    confs: List[float] = field(default_factory=list)


def get_padding_h(h: int, w: int, width_height_ratio: float, min_height: float) -> int:
    new_h = max(int(w / width_height_ratio), min_height) * 2
    padding_h = int(abs(new_h - h) / 2)
    return padding_h


def get_rotate_crop_image(img: np.ndarray, points: np.ndarray) -> np.ndarray:
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
    pts_std = np.array(
        [
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height],
        ]
    ).astype(np.float32)
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


def resize_image_within_bounds(
    img: np.ndarray, min_side_len: float, max_side_len: float
) -> Tuple[np.ndarray, float, float]:
    h, w = img.shape[:2]
    max_value = max(h, w)
    ratio_h = ratio_w = 1.0
    if max_value > max_side_len:
        img, ratio_h, ratio_w = reduce_max_side(img, max_side_len)

    h, w = img.shape[:2]
    min_value = min(h, w)
    if min_value < min_side_len:
        img, ratio_h, ratio_w = increase_min_side(img, min_side_len)
    return img, ratio_h, ratio_w


def reduce_max_side(
    img: np.ndarray, max_side_len: float = 2000
) -> Tuple[np.ndarray, float, float]:
    h, w = img.shape[:2]

    ratio = 1.0
    if max(h, w) > max_side_len:
        if h > w:
            ratio = float(max_side_len) / h
        else:
            ratio = float(max_side_len) / w

    resize_h = int(h * ratio)
    resize_w = int(w * ratio)

    resize_h = int(round(resize_h / 32) * 32)
    resize_w = int(round(resize_w / 32) * 32)

    try:
        if int(resize_w) <= 0 or int(resize_h) <= 0:
            raise ResizeImgError("resize_w or resize_h is less than or equal to 0")
        img = cv2.resize(img, (resize_w, resize_h))
    except Exception as exc:
        raise ResizeImgError() from exc

    ratio_h = h / resize_h
    ratio_w = w / resize_w
    return img, ratio_h, ratio_w


def increase_min_side(
    img: np.ndarray, min_side_len: float = 30
) -> Tuple[np.ndarray, float, float]:
    h, w = img.shape[:2]

    ratio = 1.0
    if min(h, w) < min_side_len:
        if h < w:
            ratio = float(min_side_len) / h
        else:
            ratio = float(min_side_len) / w

    resize_h = int(h * ratio)
    resize_w = int(w * ratio)

    resize_h = int(round(resize_h / 32) * 32)
    resize_w = int(round(resize_w / 32) * 32)

    try:
        if int(resize_w) <= 0 or int(resize_h) <= 0:
            raise ResizeImgError("resize_w or resize_h is less than or equal to 0")
        img = cv2.resize(img, (resize_w, resize_h))
    except Exception as exc:
        raise ResizeImgError() from exc

    ratio_h = h / resize_h
    ratio_w = w / resize_w
    return img, ratio_h, ratio_w


def add_round_letterbox(
    img: np.ndarray, padding_tuple: Tuple[int, int, int, int]
) -> np.ndarray:
    padded_img = cv2.copyMakeBorder(
        img,
        padding_tuple[0],
        padding_tuple[1],
        padding_tuple[2],
        padding_tuple[3],
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    return padded_img


class ResizeImgError(Exception):
    pass


class LangDet(Enum):
    CH = "ch"
    EN = "en"
    MULTI = "multi"


class LangCls(Enum):
    CH = "ch"


class LangRec(Enum):
    CH = "ch"
    CH_DOC = "ch_doc"
    EN = "en"
    ARABIC = "arabic"
    CHINESE_CHT = "chinese_cht"
    CYRILLIC = "cyrillic"
    DEVANAGARI = "devanagari"
    JAPAN = "japan"
    KOREAN = "korean"
    KA = "ka"
    LATIN = "latin"
    TA = "ta"
    TE = "te"


class OCRVersion(Enum):
    PPOCRV4 = "PP-OCRv4"
    PPOCRV5 = "PP-OCRv5"


class EngineType(Enum):
    ONNXRUNTIME = "onnxruntime"
    OPENVINO = "openvino"
    PADDLE = "paddle"
    TORCH = "torch"


class ModelType(Enum):
    MOBILE = "mobile"
    SERVER = "server"


class TaskType(Enum):
    DET = "det"
    CLS = "cls"
    REC = "rec"


@dataclasses.dataclass
class FileInfo:
    engine_type: EngineType
    ocr_version: OCRVersion
    task_type: TaskType
    lang_type: Enum
    model_type: ModelType