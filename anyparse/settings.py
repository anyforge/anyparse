from pydantic import BaseModel
from typing import Literal,Optional,List


class FileType(BaseModel):
    """支持的文件格式"""
    text: List = ['txt']
    html: List = ['htm','html','xhtml','shtml']
    markdown: List = ['md']
    image: List = ['apng', 'png', 'jfif', 'jpe', 'jpeg', 'jpg', 'j2c', 'j2k', 'jp2', 'jpc', 'jpf', 'jpx', 'bmp', 'webp']
    audio: List = ['mp3','wav','pcm','flac','aac','ogg','aiff']
    video: List = ['mp4','mov','m4v','avi','mkv','flv','rmvb','wmv']
    pdf: List = ['pdf']
    docx: List = ['doc','docx']
    pptx: List = ['ppt','pptx']
    table: List = ['csv','xls','xlsx']
    code: List = ['c', 'cpp', 'h', 'hpp', 'cc', 'hh', 'c++', 'h++', 'cxx', 'hxx', 'py', 'java', 'scala', 'js', 'css', 'scss', 
                  'm', 'sql', 'conf', 'ini', 'sh', 'zsh', 'ksh', 'bash', 'bashrc', 'zshrc', 'kshrc', 'pl', 'jsp', 'jsx', 'go', 
                  'graphql', 'gv', 'dot', 'groovy', 'gradle', 'ini', 'cfg', 'inf', 'k', 'less', 'lua', 'makefile', 'def', 'mod', 
                  'mojo', '🔥', 'perl', 'php', 'py', 'pyw', 'pyi', 'r', 'sass', 'scala', 'scss', 'sql', 'swift', 'service', 'socket', 
                  'device', 'mount', 'toml', 'tcsh', 'csh', 'sql', 'tt', 'ttl', 'ts', 'vbs', 'vb', 'vp', 'vim', 'vimrc', 'gvimrc', 'cl', 
                  'ph', 'toc', 'xml', 'yaml', 'yml', 'yang'
                ]


class Settings(BaseModel):
    class Config:
        extra = 'allow'  # 允许额外的字段
        arbitrary_types_allowed = True #允许 Pydantic 处理任意类型
    
    ocr_mode: Literal["base","plus"] = "plus"
    pdf_mode: Literal["auto","txt", "ocr"] = "auto"
    
    ### ocr
    ocr_config_path: str = "config/anyocr.yaml"

    ## table
    table_chunk_size: Optional[int] = None
    table_encoding: str = 'utf-8'
    
    ## html
    html_encoding: str = 'utf-8'
    html_strip: bool = False
    html_ignore_links: bool = False  # 忽略链接
    html_ignore_images: bool = False # 忽略图片
    html_ignore_emphasis: bool = False # 忽略强调（如加粗、斜体
    html_body_width: int =78  # 不限制输出宽度
    html_unicode_snob: bool = False  # 始终使用 Unicode 字符
    
    ## code
    code_encoding: str = 'utf-8'
    
    ## markdown
    markdown_encoding: str = 'utf-8'
    
    ## text
    text_encoding: str = 'utf-8'
    
    ## pdf 是否分栏
    is_column: bool = False
