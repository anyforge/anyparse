from itertools import chain
from typing import List, Optional
from rich import print as _print
from pathlib import Path
from .base import AnyDataModel, Field, model_validator
from ..callbacks.base import BaseCallback
from ..callbacks.parsecallback import ParseCallback


print = _print

# 获取用户主目录，然后拼接 '.cache'
default_cache_dir = Path.home() / '.cache/anyparse'
default_maxsize = 50.0
default_file_read_size = 1024*1024*2



class FileTypes(AnyDataModel):
    """支持的文件格式"""
    text: List = ['txt']
    html: List = ['htm','html','xhtml','shtml']
    markdown: List = ['md', 'rst']
    image: List = ['png', 'jpeg', 'jpg', 'webp']
    pdf: List = ['pdf']
    epub: List = ['epub']
    office: List = ['docx','pptx','xlsx']
    csv: List = ['tsv','csv']
    ipynb: List = ['ipynb']
    email: List = ['eml']
    audio: List = ['wav', 'mp3', 'aac', 'flac']
    video: List = ['mp4','mov','mkv','webm','avi']

    def total_dict(self) -> dict:
        """获取当前实例实际持有的文件格式 (基于实例属性)"""
        # model_dump() 会把实例的字段名和实际值打包成字典返回
        return self.model_dump()
    
    def total_list(self) -> List[str]:
        """获取当前实例实际持有的文件格式 (基于实例属性)"""
        return list(chain.from_iterable(self.model_dump().values()))
    

class MimeTypes(AnyDataModel):
    txt: str = "data:text/plain;base64,"
    htm: str = "data:text/html;base64,"
    html: str = "data:text/html;base64,"
    xhtml: str = "data:text/html;base64,"
    shtml: str = "data:text/html;base64,"
    md: str = "data:text/markdown;base64,"
    rst: str = "data:text/rst;base64,"
    png: str = "data:image/png;base64,"
    jpeg: str = "data:image/jpeg;base64,"
    jpg: str = "data:image/jpeg;base64,"
    webp: str = "data:image/webp;base64,"
    pdf: str = "data:application/pdf;base64,"
    epub: str = "data:application/epub;base64,"
    docx: str = "data:application/docx;base64,"
    pptx: str = "data:application/pptx;base64,"
    xlsx: str = "data:application/xlsx;base64,"
    tsv: str = "data:text/csv;base64,"
    csv: str = "data:text/csv;base64,"
    ipynb: str = "data:application/ipynb;base64,"
    eml: str = "data:message/eml;base64,"
    wav: str = "data:audio/wav;base64,"
    mp3: str = "data:audio/mpeg;base64,"
    aac: str = "data:audio/aac;base64,"
    flac: str = "data:audio/flac;base64,"
    mp4: str = "data:video/mp4;base64,"
    mov: str = "data:video/mov;base64,"
    mkv: str = "data:video/mkv;base64,"
    webm: str = "data:video/webm;base64,"
    avi: str = "data:video/avi;base64,"    


class Settings(AnyDataModel):
    @model_validator(mode="after")
    def convert_to_path(self):
        self.cache_dir = Path(self.cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self
    
    cache_dir: str | Path = Field(
        default = default_cache_dir,
        description = "缓存目录"
    )
    parse_callback: BaseCallback = Field(
        default = ParseCallback(),
        description = "回调函数"
    )
    autocal_md5: Optional[bool] = Field(
        default = True,
        description = "是否自动计算md5"
    )
    autodetect_encoding: Optional[bool] = Field(
        default = True,
        description = "是否自动检测编码"
    )
        
    use_doc_cls: Optional[bool] = Field(
        default = False,
        description = "是否对文档进行方向矫正"
    )
    use_doc_rectifier: Optional[bool] = Field(
        default = False,
        description = "是否对文档进行矫正"
    )
    use_doc_layout: Optional[bool] = Field(
        default = True,
        description = "是否对文档进行布局识别"
    )
    doc_layout_image_min_size: Optional[int] = Field(
        default = 400,
        description = "文档布局最小尺寸"
    )
    use_image_resize: Optional[bool] = Field(
        default = False,
        description = "是否对图片进行缩放"
    )
    
    text_encoding: Optional[str] = Field(
        default = "utf-8",
        description = "文本编码"
    )
    table_chunk_size: Optional[int] = Field(
        default = None,
        description = "表格分块大小"
    )
    table_custom_separator: Optional[str] = Field(
        default = None,
        description = "表格自定义分隔符"
    )
    table_sniffer: Optional[bool] = Field(
        default = True,
        description = "表格自动检测分隔符"
    )
    
    dpi: Optional[int] = Field(
        default = 200,
        description = "图片分辨率"
    )
    verbose: Optional[bool] = Field(
        default = True,
        description = "是否详细输出"
    )
    stream: Optional[bool] = Field(
        default = False,
        description = "是否流式处理"
    )
    docx_extract_headers_footers: Optional[bool] = Field(
        default = True,
        description = "是否提取docx页眉页脚"
    )
    docx_extract_images: Optional[bool] = Field(
        default = True,
        description = "是否提取docx图片"
    )
    pptx_extract_images: Optional[bool] = Field(
        default = True,
        description = "是否提取pptx图片"
    )
    excel_extract_images: Optional[bool] = Field(
        default = True,
        description = "是否提取excel图片"
    )
    excel_max_rows: Optional[int] = Field(
        default = None,
        description = "excel最大行数"
    )
    
    max_new_tokens: Optional[int] = Field(
        default = 8192,
        description = "最大生成长度"
    )
    
    image_batch_size: Optional[int] = Field(
        default = 1,
        description = "图片批量处理大小"
    )
    ocr_batch_size: Optional[int] = Field(
        default = 1,
        description = "文本批量处理大小"
    )
    asr_batch_size: Optional[int] = Field(
        default = 1,
        description = "语音批量处理大小"
    )
    
    doc_cls: Optional[dict] = Field(
        default = {
            "model_path": "resource/models/pp_lcnet_x1_0_doc_ori",
            'batch_size': 1,
            'dtype': 'auto',
            'device_map': 'auto',
        },
        description = "分类模型"
    )
 
    doc_rectifier: Optional[dict] = Field(
        default = {
            "model_path": "resource/models/pp_uvdoc",
            'batch_size': 1,
            'dtype': 'auto',
            'device_map': 'auto',
        },
        description = "文档扭曲矫正模型"
    )   
    
    layout: Optional[dict] = Field(
        default = {
            "model_path": "resource/models/ppdoclayout-v3",
            "threshold": 0.3,
            "batch_size": 1,
            "dtype": "auto",
            "device_map": "auto",
            "layout_nms": True
        },
        description = "布局模型"
    )
    vlm: Optional[dict] = Field(
        default = {
            'model_type': 'vllm',
            "paddleocr": {
                "model_class": "PaddleOCRClient.PPOCRV6",
                "model_path": "resource/models/paddleocrv6-small",
                "batch_size": 6,
                'dtype': 'auto',
                'device_map': 'auto',
            },
            'glmocr_v1': {
                'model_class': 'GlmOCRClient.GLMOCRV1',
                'model_path': 'resource/models/glmocr-v1',
                'batch_size': 1,
                'max_new_tokens': 8192,
                'dtype': 'auto',
                'device_map': 'auto',
                'min_pixels': 12544,
                'max_pixels': 71372800
            },
            "paddleocrvl": {
                'model_calss': 'PaddleOCRVLClient.PPOCRVLClient',
                'model_path': 'resource/models/paddleocrvl-v1.6',
                'batch_size': 1,
                'max_new_tokens': 8192,
                'dtype': 'auto',
                'device_map': 'auto',   
                'attn_implementation': None,  
                "truncate_content": True,
                "truncate_content_list": [5000, 50],         
            },
            "vllm": {
                "model_class": 'OpenAIClient.OpenAIClient',
                "base_url": "http://localhost:18003/v1",
                "api_key": "sk-123456",
                "model": "PaddleOCR-VL-1.6",
                "stream": False,
                "timeout": 1800.0,
                "max_retries": 2,
                "batch_size": 8,
                "max_new_tokens": 8192,
                "task_prompt_map": {
                    "abstract": "OCR:",
                    "algorithm": "OCR:",
                    "content": "OCR:",
                    "doc_title": "OCR:",
                    "figure_title": "OCR:",
                    "paragraph_title": "OCR:",
                    "reference_content": "OCR:",
                    "text": "OCR:",
                    "vertical_text": "OCR:",
                    "vision_footnote": "OCR:",
                    "seal": "OCR:",
                    "formula_number": "OCR:",
                    "header": "OCR:",
                    "footer": "OCR:",
                    "number": "OCR:",
                    "footnote": "OCR:",
                    "aside_text": "OCR:",
                    "reference": "OCR:",
                    "footer_image": "OCR:",
                    "header_image": "OCR:",
                    "image": "OCR:",
                    "table": "Table Recognition:",
                    "display_formula": "Formula Recognition:",
                    "inline_formula": "Formula Recognition:",
                    "chart": "Chart Recognition:"
                },
                "client_args": {
                },
                "call_args": {
                },
                "prompt_template": """
                    [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url", 
                                    "image_url": {
                                        "url": "{{ data_url }}"
                                    }
                                },
                                {
                                    "type": "text", 
                                    "text": "{{ prompt }}"
                                }
                            ]
                        }
                    ]
                """
            }
        },
        description = "ocr模型"
    )
    converters_models: Optional[List[dict]] = Field(
        default = [
            {"name": "markdown", "model_class": "mkd.MkdConverter"},
            {"name": "text", "model_class": "text.TextConverter"},
            {"name": "csv", "model_class": "csv.CsvConverter"},
            {"name": "html", "model_class": "html.HtmlConverter"},
            {"name": "pdf", "model_class": "pdf.PdfConverter"},
            {"name": "epub", "model_class": "epub.EpubConverter"},
            {"name": "ipynb", "model_class": "ipynb.IpynbConverter"},
            {"name": "email", "model_class": "email.EmailConverter"},
            {"name": "docx", "model_class": "office_mineru.docx.DocxConverter"},
            {"name": "pptx", "model_class": "office_mineru.pptx.PptxConverter"},
            {"name": "xlsx", "model_class": "office_mineru.xlsx.XlsxConverter"},
        ],
        description = "转换器注册表"
    )
