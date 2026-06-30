import re
import time
import traceback
import threading
from contextlib import contextmanager
from typing import Any, Dict, List, Literal
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from ..schemas.base import AnyDataModel


class PdfPageOutput(AnyDataModel):
    page_idx: int = 0
    reason: str = ''
    label: Literal["txt", "ocr"] = "ocr"
    elapse_times: float = 0.0


class PdfClassifier(object):
    _pdfium_lock = threading.RLock()
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        # --- 配置区 ---
        self.chars_threshold = self.kwargs.get(
            "chars_threshold", 
            50
        )  # 每页最少字符数
        self.high_image_coverage_threshold = self.kwargs.get(
            "high_image_coverage_threshold", 
            0.8
        )  # 图像覆盖超过 80% 判定为图  high_image_coverage_threshold
        self.text_quality_bad_threshold = self.kwargs.get(
            "text_quality_bad_threshold", 
            0.03
        )  # 文本质量差的阈值 text_quality_bad_threshold
        self.max_page_aspect_ratio = self.kwargs.get(
            "max_page_aspect_ratio", 
            10.0
        )  # 极端宽高比 max_page_aspect_ratio

        # --- 工具函数与锁 ---
        self.allow_control_codes = self.kwargs.get(
            "allow_control_codes", 
            {9, 10, 13}
        )  # 允许的控制字符：制表符、换行、回车 allow_control_codes
        self.private_use_area_start = self.kwargs.get(
            "private_use_area_start", 
            0xE000
        )   # 私有使用区域起始码 private_use_area_start
        self.private_use_area_end = self.kwargs.get(
            "private_use_area_end", 
            0xF8FF
        )    # 私有使用区域结束码 private_use_area_end

    @contextmanager
    def _pdfium_guard(self):
        """上下文管理器，用于线程安全"""
        with self._pdfium_lock:
            yield

    def _is_disallowed_control_unicode(self, unicode_code: int) -> bool:
        """检查是否为非法控制字符（排除允许的空白字符）"""
        flag1 = (0 <= unicode_code < 32 or 127 <= unicode_code <= 159)
        flag2 = unicode_code not in self.allow_control_codes
        return (flag1 and flag2)

    def _analyze_page(self, page: pdfium.PdfPage, page_index: int) -> Dict[str, Any]:
        """
        分析单页 PDF。
        Args:
            page: pypdfium2 的页面对象
            page_index: 页码索引
        Returns:
            Dict: 包含 result 和 reason 的结果
        """
        try:
            # 1. 检查页面宽高比 (极端比例通常是大图)
            page_width, page_height = page.get_size()
            if page_width <= 0 or page_height <= 0:
                return {"result": "ocr", "reason": "invalid_size"}
            aspect_ratio = max(page_width / page_height, page_height / page_width)
            if aspect_ratio > self.max_page_aspect_ratio:
                return {"result": "ocr", "reason": "extreme_aspect_ratio"}

            # 2. 检查文本覆盖率 (是否有足够文字)
            text_page = page.get_textpage()
            text = text_page.get_text_bounded()
            cleaned_text = re.sub(r"\s+", "", text)
            if len(cleaned_text) < self.chars_threshold:
                return {"result": "ocr", "reason": "insufficient_text"}

            # 3. 检查文本质量 (乱码检测)
            char_count = text_page.count_chars()
            abnormal_chars = 0
            for i in range(char_count):
                unicode_code = pdfium_c.FPDFText_GetUnicode(text_page, i)
                if (unicode_code == 0 or 
                    unicode_code == 0xFFFD or 
                    self._is_disallowed_control_unicode(unicode_code) or
                    (self.private_use_area_start <= unicode_code <= self.private_use_area_end)):
                    abnormal_chars += 1
                    
            if char_count > 0:
                abnormal_ratio = abnormal_chars / char_count
                if abnormal_ratio >= self.text_quality_bad_threshold:
                    return {"result": "ocr", "reason": "poor_text_quality"}

            # 4. 检查图像覆盖 (满屏大图)
            page_bbox = page.get_bbox()
            page_area = abs((page_bbox[2] - page_bbox[0]) * (page_bbox[3] - page_bbox[1]))
            image_area = 0.0
            
            # 遍历页面对象，统计图像面积
            for obj in page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE], max_depth=3):
                left, bottom, right, top = obj.get_pos()
                image_area += max(0.0, right - left) * max(0.0, top - bottom)
                
            if page_area > 0:
                coverage_ratio = image_area / page_area
                if coverage_ratio >= self.high_image_coverage_threshold:
                    return {"result": "ocr", "reason": "high_image_coverage"}

            # 5. 如果以上都没触发，判定为文本
            return {"result": "txt", "reason": "valid_text"}
            
        except:
            traceback.print_exc()
            return {"result": "ocr", "reason": f"analysis_error"}

    def invoke(self, page: pdfium.PdfPage, page_index: int = 0) -> PdfPageOutput:
        """
        入口方法：接收一个 PdfPage 对象，返回分类结果。
        这个方法是线程安全的。
        """
        start_time = time.perf_counter()
        
        # 只在核心分析时加锁
        with self._pdfium_guard():
            res = self._analyze_page(page, page_index)
            
        end_time = time.perf_counter()
        output = PdfPageOutput(
            page_idx = page_index,
            reason = res.get('reason', '').lower(),
            label = res.get('result', '').lower(),
            elapse_times = end_time - start_time,
        )
        return output