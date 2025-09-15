import cv2
import math
import time
import dataclasses
import numpy as np
from pathlib import Path
from typing import Any, List, Dict, Optional, Tuple, Union
from .utils import WordInfo, WordType,has_chinese_char
from .engine import FileInfo, get_engine


@dataclasses.dataclass
class TextRecConfig:
    intra_op_num_threads: int = -1
    inter_op_num_threads: int = -1
    use_cuda: bool = False
    use_dml: bool = False
    model_path: Union[str, Path, None] = None

    rec_batch_num: int = 6
    rec_img_shape: Tuple[int, int, int] = (3, 48, 320)
    rec_keys_path: Union[str, Path, None] = None


@dataclasses.dataclass
class TextRecInput:
    img: Union[np.ndarray, List[np.ndarray], None] = None
    return_word_box: bool = False


@dataclasses.dataclass
class TextRecOutput:
    imgs: Optional[List[np.ndarray]] = None
    txts: Optional[Tuple[str]] = None
    scores: Tuple[float] = (1.0,)
    word_results: Tuple[Tuple[str, float, Optional[List[List[int]]]]] = (
        ("", 1.0, None),
    )
    elapse: Optional[float] = None
    lang_type: Optional[str] = None

    def __len__(self):
        if self.txts is None:
            return 0
        return len(self.txts)
    
    
class CTCLabelDecode:
    def __init__(
        self,
        character: Optional[List[str]] = None,
        character_path: Union[str, Path, None] = None,
    ):
        self.character = self.get_character(character, character_path)
        self.dict = {char: i for i, char in enumerate(self.character)}

    def __call__(
        self, preds: np.ndarray, return_word_box: bool = False, **kwargs
    ) -> Tuple[List[Tuple[str, float]], List[Any]]:
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)

        wh_ratio_list = kwargs.get("wh_ratio_list", (1.0,))
        max_wh_ratio = kwargs.get("max_wh_ratio", 1.0)

        line_results, word_results = self.decode(
            preds_idx,
            preds_prob,
            return_word_box,
            wh_ratio_list,
            max_wh_ratio,
            remove_duplicate=True,
        )
        return line_results, word_results

    def get_character(
        self,
        character: Optional[List[str]] = None,
        character_path: Union[str, Path, None] = None,
    ) -> List[str]:
        if character is None and character_path is None:
            raise ValueError("character must not be None")

        character_list = None
        if character:
            character_list = character

        if character is None and character_path is not None:
            character_list = self.read_character_file(character_path)

        if character_list is None:
            raise ValueError("character must not be None")

        character_list = self.insert_special_char(
            character_list, " ", len(character_list)
        )
        character_list = self.insert_special_char(character_list, "blank", 0)
        return character_list

    @staticmethod
    def read_character_file(character_path: Union[str, Path]) -> List[str]:
        character_list = []
        with open(character_path, "rb") as f:
            lines = f.readlines()
            for line in lines:
                line = line.decode("utf-8").strip("\n").strip("\r\n")
                character_list.append(line)
        return character_list

    @staticmethod
    def insert_special_char(
        character_list: List[str], special_char: str, loc: int = -1
    ) -> List[str]:
        character_list.insert(loc, special_char)
        return character_list

    def decode(
        self,
        text_index: np.ndarray,
        text_prob: Optional[np.ndarray] = None,
        return_word_box: bool = False,
        wh_ratio_list: Tuple[float] = (1.0,),
        max_wh_ratio: float = 1.0,
        remove_duplicate: bool = False,
    ) -> Tuple[List[Tuple[str, float]], List[WordInfo]]:
        result_list, result_words_list = [], []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            token_indices = text_index[batch_idx]

            selection = np.ones(len(token_indices), dtype=bool)
            if remove_duplicate:
                selection[1:] = token_indices[1:] != token_indices[:-1]

            for ignored_token in ignored_tokens:
                selection &= token_indices != ignored_token

            if text_prob is not None:
                conf_list = np.array(text_prob[batch_idx][selection]).tolist()
                conf_list = [round(conf, 5) for conf in conf_list]
            else:
                conf_list = [1] * len(selection)

            if len(conf_list) == 0:
                conf_list = [0]

            char_list = [
                self.character[text_id] for text_id in token_indices[selection]
            ]
            text = "".join(char_list)

            result_list.append((text, np.mean(conf_list).round(5).tolist()))

            if return_word_box:
                rec_word_info = self.get_word_info(text, selection)
                rec_word_info.line_txt_len = (
                    len(token_indices) * wh_ratio_list[batch_idx] / max_wh_ratio
                )
                rec_word_info.confs = conf_list
                result_words_list.append(rec_word_info)
        return result_list, result_words_list

    @staticmethod
    def get_word_info(text: str, selection: np.ndarray) -> WordInfo:
        """
        Group the decoded characters and record the corresponding decoded positions.
        from https://github.com/PaddlePaddle/PaddleOCR/blob/fbba2178d7093f1dffca65a5b963ec277f1a6125/ppocr/postprocess/rec_postprocess.py#L70
        """
        word_list = []
        word_col_list = []
        state_list = []

        word_content = []
        word_col_content = []

        valid_col = np.where(selection)[0]
        if len(valid_col) <= 0:
            return WordInfo()

        col_width = np.zeros(valid_col.shape)
        col_width[1:] = valid_col[1:] - valid_col[:-1]
        col_width[0] = min(3 if has_chinese_char(text[0]) else 2, int(valid_col[0]))

        state = None
        for c_i, char in enumerate(text):
            if char.isspace():
                if word_content:
                    word_list.append(word_content)
                    word_col_list.append(word_col_content)
                    state_list.append(state)
                    word_content = []
                    word_col_content = []
                continue

            c_state = WordType.CN if has_chinese_char(char) else WordType.EN_NUM
            if state is None:
                state = c_state

            if state != c_state or col_width[c_i] > 5:
                if len(word_content) != 0:
                    word_list.append(word_content)
                    word_col_list.append(word_col_content)
                    state_list.append(state)
                    word_content = []
                    word_col_content = []
                state = c_state

            word_content.append(char)
            word_col_content.append(int(valid_col[c_i]))

        if len(word_content) != 0:
            word_list.append(word_content)
            word_col_list.append(word_col_content)
            state_list.append(state)

        return WordInfo(words=word_list, word_cols=word_col_list, word_types=state_list)

    @staticmethod
    def get_ignored_tokens() -> List[int]:
        return [0]  # for ctc blank   


class TextRecognizer:
    def __init__(self, cfg: Dict[str, Any]):
        self.session = get_engine(cfg.engine_type)(cfg)

        # onnx has inner character, other engine get or download character_dict_path
        character, character_dict_path = self.get_character_dict(cfg)

        self.postprocess_op = CTCLabelDecode(
            character=character, character_path=character_dict_path
        )

        self.rec_batch_num = cfg["rec_batch_num"]
        self.rec_image_shape = cfg["rec_img_shape"]

    def get_character_dict(self, cfg):
        character = None
        dict_path = cfg.get("rec_keys_path", None)
        if self.session.have_key():
            character = self.session.get_character_list()
        return character, dict_path

    def __call__(self, args: TextRecInput) -> TextRecOutput:
        img_list = [args.img] if isinstance(args.img, np.ndarray) else args.img
        return_word_box = args.return_word_box

        width_list = [img.shape[1] / float(img.shape[0]) for img in img_list]

        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))

        img_num = len(img_list)
        rec_res = [("", 0.0)] * img_num

        batch_num = self.rec_batch_num
        elapse = 0
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)

            # Parameter Alignment for PaddleOCR
            imgC, imgH, imgW = self.rec_image_shape[:3]
            max_wh_ratio = imgW / imgH
            wh_ratio_list = []
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
                wh_ratio_list.append(wh_ratio)

            norm_img_batch = []
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]], max_wh_ratio)
                norm_img_batch.append(norm_img[np.newaxis, :])
            norm_img_batch = np.concatenate(norm_img_batch).astype(np.float32)

            start_time = time.perf_counter()
            preds = self.session(norm_img_batch)
            line_results, word_results = self.postprocess_op(
                preds,
                return_word_box,
                wh_ratio_list=wh_ratio_list,
                max_wh_ratio=max_wh_ratio,
            )

            for rno, one_res in enumerate(line_results):
                if return_word_box:
                    rec_res[indices[beg_img_no + rno]] = (one_res, word_results[rno])
                    continue

                rec_res[indices[beg_img_no + rno]] = (one_res, None)
            elapse += time.perf_counter() - start_time

        all_line_results, all_word_results = list(zip(*rec_res))
        txts, scores = list(zip(*all_line_results))
        return TextRecOutput(img_list, txts, scores, all_word_results, elapse)

    def resize_norm_img(self, img: np.ndarray, max_wh_ratio: float) -> np.ndarray:
        img_channel, img_height, img_width = self.rec_image_shape
        assert img_channel == img.shape[2]

        img_width = int(img_height * max_wh_ratio)

        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(img_height * ratio) > img_width:
            resized_w = img_width
        else:
            resized_w = int(math.ceil(img_height * ratio))

        resized_image = cv2.resize(img, (resized_w, img_height))
        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5

        padding_im = np.zeros((img_channel, img_height, img_width), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im
