import re
import time
import pymupdf
import traceback
from typing import Literal
from pydantic import BaseModel


class pdfClassifyOutput(BaseModel):
    clean_text_len: int = 0
    suspicious_ratio: float = 0.0
    coverage_ratio: float = 0.0
    elapse_times: float = 0.0
    label: Literal["txt", "ocr"] = "ocr"


class pdfClassify(object):
    def __init__(
        self, 
        min_chars_num: int = 50,  ### 最少字符数
        suspicious_ratio: float = 0.05, ### 页面文字乱码率
        coverage_ratio: float = 0.8, ### 图片页面覆盖率
    ):
        self.min_chars_num = min_chars_num
        self.suspicious_ratio = suspicious_ratio
        self.coverage_ratio = coverage_ratio
        
    def invoke_textnum(self, page: pymupdf.Page, **kwargs):
        min_chars_num = int(kwargs.get("min_chars_num", self.min_chars_num))
        page_text = page.get_text()
        page_text_len = len(page_text)
        if page_text_len <= min_chars_num:
            return True, page_text_len
        else:
            cleaned_text = re.sub(r'\s+', '', page_text)
            cleaned_total_chars = len(cleaned_text)
            if cleaned_total_chars < min_chars_num:
                return True, cleaned_total_chars
            else:
                return False, cleaned_total_chars
            
    def invoke_suspicious(self, page: pymupdf.Page, **kwargs):
        min_chars_num = int(kwargs.get("min_chars_num", self.min_chars_num))
        suspicious_ratio_score = float(kwargs.get("suspicious_ratio", self.suspicious_ratio))
        page_text = page.get_text()
        page_text_len = len(page_text)
        if page_text_len <= min_chars_num:
            return True, 1.0       
        # 1. 检测Unicode替换字符 (U+FFFD)
        replacement_chars = page_text.count('\ufffd')
        replacement_ratio = replacement_chars / page_text_len
        # 2. 检测疑似乱码字符模式
        # 检测连续的问号或特殊字符
        suspicious_patterns = [
            r'\?{3,}',  # 连续3个以上问号
            r'[^\x00-\x7F\u4e00-\u9fff\u0080-\u00FF\u3000-\u303F\uff00-\uffef]{5,}',  # 连续非常见字符
        ]
        
        suspicious_count = 0
        suspicious_len = 0
        for pattern in suspicious_patterns:
            matches = re.findall(pattern, page_text)
            matches = [x for x in matches]
            suspicious_len += len(matches)
            suspicious_count += sum(len(x) for x in matches)
            
        suspicious_ratio = suspicious_count / (suspicious_count + page_text_len - suspicious_len) 
        min_ratio = float(min(replacement_ratio, suspicious_ratio))     
        if min_ratio > suspicious_ratio_score:
            return True, min_ratio
        else:
            return False, min_ratio
        
    def invoke_coverage(self, page: pymupdf.Page, **kwargs):
        coverage_ratio_score = float(kwargs.get("coverage_ratio", self.coverage_ratio))
        page_rect = page.rect
        page_area = page_rect.width * page_rect.height
        image_list = page.get_images(full=True)
        image_area = 0.0
        for idx,image in enumerate(image_list):
            try:
                image_rects = page.get_image_rects(image[0])
                for rect in image_rects:
                    img_area = rect.width * rect.height
                    image_area += img_area
            except:
                traceback.print_exc()
                continue
        # 计算覆盖率
        coverage_ratio = min(image_area / page_area, 1.0) if page_area > 0.0 else 0.0
        if coverage_ratio > coverage_ratio_score:
            return True, coverage_ratio
        else:
            return False, coverage_ratio
        
    def invoke(self, page: pymupdf.Page, pdf_mode: str = "auto", **kwargs):
        try:
            if pdf_mode == "auto":
                tt = time.time()
                clean_total_flag, cleaned_total_chars = self.invoke_textnum(page, **kwargs)
                min_ratio_flag, min_ratio = self.invoke_suspicious(page, **kwargs)
                coverage_ratio_flag, coverage_ratio = self.invoke_coverage(page, **kwargs)
                if clean_total_flag or min_ratio_flag:
                    label = "ocr"
                else:
                    if coverage_ratio_flag:
                        label = "ocr"
                    else:
                        label = "txt"
                tt = time.time() - tt
                output = pdfClassifyOutput(
                    clean_text_len = int(cleaned_total_chars),
                    suspicious_ratio = float(min_ratio),
                    coverage_ratio = float(coverage_ratio),
                    elapse_times = tt,
                    label = label                
                )
            elif pdf_mode == "txt":
                output = pdfClassifyOutput(label = "txt")
            elif pdf_mode == "ocr":
                output = pdfClassifyOutput(label = "ocr")
            else:
                output = pdfClassifyOutput(label = "ocr")
        except:
            traceback.print_exc()
            output = pdfClassifyOutput(label = "ocr")
        finally:
            return output