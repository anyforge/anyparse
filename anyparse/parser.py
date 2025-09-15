import warnings
warnings.filterwarnings("ignore")
import time
import datetime
from pathlib import Path
from .anyocr.parser import AnyOCR
from .converters.html_converter import htmlConverter
from .converters.text_converter import (
    textConverter,
    markdownConverter
)
from .converters.office_converter import (
    docxConverter,
    pptxConverter,
    tableConverter
)
from .converters.pdf_converter import pdfConverter
from .constructs.callback import parseCallback
from .constructs.detector import detect_file_encodings
from .constructs.document import Metadata, Pagedata, AnyParseOutput, AnyParseStream
from .utils.utils import autodlhash, format_filesize, create_uuid
from .settings import Settings, FileType


class AnyParse(object):
    """
    解析各种文件为markdown
    """
    def __init__(self, config = {}):
        self.config = config
        self.settings = Settings(**self.config)
        self.file_type_instance = FileType(**self.config.get('filtypes', {}))
        self.load_all_parsers()
        
    def load_all_parsers(self):
        ### load parsers
        self.markdownconverter = markdownConverter()
        self.textconverter = textConverter()
        self.htmlconverter = htmlConverter()
        self.docxconverter = docxConverter()
        self.pptxconverter = pptxConverter()
        self.tableconverter = tableConverter()
        self.pdflconverter = pdfConverter()
        
        ### load models
        self.ocrmodel = AnyOCR(config_path=self.settings.ocr_config_path)
        return    
    
    def check_health(self):
        return True
    
    def invoke(self,file='',fileid = '', cal_file_md5 = False, autodetect_encoding = True, parsecallback = None, **kwargs):
        if not parsecallback:
            parsecallback = parseCallback()
        processfile = Path(file)
        processfilename = processfile.name
        processfiletype = processfile.suffix.strip('.')
        processfilehash = ''
        if cal_file_md5:
            processfilehash = autodlhash().encrypt_file(processfile)
        processfilesize = format_filesize(processfile.stat().st_size)
        ### start process file
        metadata = Metadata(
            file_type = processfiletype,
            file_name = processfilename,
            file_md5 = processfilehash,
            file_id = fileid if fileid else create_uuid(),
            file_size = f"{processfilesize}",            
        )
        document = AnyParseOutput(
            metas = metadata,
            pages = [],
            elapse_times = {}
        )   
        pagescontentlist = []
        ocr_mode = kwargs.pop("ocr_mode", self.settings.ocr_mode)
        pdf_mode = kwargs.pop("pdf_mode", self.settings.pdf_mode)
        elapse_time = time.time()
        match processfiletype.lower():
            case ft if ft in self.file_type_instance.text:
                text_encoding = kwargs.pop('text_encoding',self.settings.text_encoding)
                if autodetect_encoding:
                    encodings = detect_file_encodings(processfile,file_type="text")
                    if encodings:
                        text_encoding = encodings
                fileres = self.textconverter.invoke_text(
                    processfile,
                    encoding = text_encoding,
                    **kwargs                        
                )
                for idx,line in enumerate(fileres):
                    pagecontent = Pagedata(
                        pageid = f"{idx+1}",
                        meta = line['type'],
                        content = line['content']
                    )
                    pagescontentlist.append(pagecontent)
                                    
            case ft if ft in self.file_type_instance.markdown:
                markdown_encoding = kwargs.pop('markdown_encoding',self.settings.markdown_encoding)
                if autodetect_encoding:
                    encodings = detect_file_encodings(processfile,file_type="text")
                    if encodings:
                        markdown_encoding = encodings
                fileres = self.markdownconverter.invoke_markdown(
                    processfile,
                    encoding = markdown_encoding,
                    **kwargs                      
                )
                for idx,line in enumerate(fileres):
                    linecontent = line['content']
                                                     
                    pagecontent = Pagedata(
                        pageid = f"{idx+1}",
                        meta = line['type'],
                        content = linecontent,
                    )
                    pagescontentlist.append(pagecontent)                   
                
            case ft if ft in self.file_type_instance.html:
                html_encoding = kwargs.pop('html_encoding',self.settings.html_encoding)
                if autodetect_encoding:
                    encodings = detect_file_encodings(processfile,file_type="text")
                    if encodings:
                        html_encoding = encodings   
                fileres = self.htmlconverter.invoke_html(
                    processfile,
                    encoding = html_encoding,
                    **kwargs                   
                )   
                for idx,line in enumerate(fileres):
                    linecontent = line['content']
                                                     
                    pagecontent = Pagedata(
                        pageid = f"{idx+1}",
                        meta = line['type'],
                        content = linecontent,
                    )
                    pagescontentlist.append(pagecontent)          
                
            case ft if ft in self.file_type_instance.image:
                stream = kwargs.pop('stream', False)
                if not stream:
                    fileres = self.pdflconverter.invoke_image(
                        processfile,
                        self.ocrmodel,
                        ocr_mode = ocr_mode,
                        **kwargs,
                    )    
                    for idx,line in enumerate(fileres):
                        pagecontent = Pagedata(
                            pageid = f"{idx+1}",
                            meta = line['type'],
                            content = line['content']
                        )
                        pagescontentlist.append(pagecontent)    
                else: 
                    async def image_parse_stream():
                        elapse_time = time.time()       
                        fileres = await self.pdflconverter.ainvoke_image_stream(
                            processfile, 
                            self.ocrmodel, 
                            **kwargs
                        )    
                        async for item in fileres:
                            try:
                                output = AnyParseStream(
                                    pageid = "1",
                                    metas = metadata,
                                    content = item.content,
                                    elapse_times = item.elapse_times
                                )                            
                            except:
                                output = AnyParseStream(
                                    pageid = "1",
                                    metas = metadata,
                                    content = "",
                                    elapse_times={
                                        "datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    }
                                )
                                
                            finally:
                                if len(output.content) > 0:
                                    yield output
                        elapse_time = time.time() - elapse_time
                        elapse_time = {
                            "total": elapse_time
                        }
                        output = AnyParseStream(
                            pageid = "1",
                            metas = metadata,
                            content = "",
                            elapse_times = elapse_time
                        )
                        yield output
                    return image_parse_stream()
            
            case ft if ft in self.file_type_instance.pdf:
                is_column = kwargs.pop("is_column", self.settings.is_column)
                stream = kwargs.pop('stream', False)
                if not stream:
                    fileres = self.pdflconverter.invoke_pdf(
                        processfile,
                        self.ocrmodel,
                        pdf_mode = pdf_mode,
                        ocr_mode = ocr_mode,
                        parsecallback = parsecallback,
                        is_column = is_column,
                        **kwargs,
                    )                                        
                    for idx,line in enumerate(fileres):
                        pagecontent = Pagedata(
                            pageid = f"{line['pageid']}",
                            meta = line['type'],
                            content = line['content'],
                            # images = line['images']
                        )
                        pagescontentlist.append(pagecontent)    
                else:
                    async def pdf_parse_stream():
                        elapse_time = time.time()       
                        fileres = self.pdflconverter.ainvoke_pdf_stream(
                            processfile, 
                            self.ocrmodel, 
                            parsecallback = parsecallback,
                            **kwargs
                        )    
                        pageid = 1
                        async for item in fileres:
                            # yield f"[PAGEDONE]:{page_number + 1}"
                            if isinstance(item, str):
                                if item.startswith("[PAGEDONE]:"):
                                    newpageid = int(item.split(":")[1])
                                    if newpageid != pageid:
                                        output = AnyParseStream(
                                            pageid = f"{pageid}",
                                            metas = metadata,
                                            content = "\n\n",
                                            elapse_times={
                                                "datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                            }
                                        )
                                        yield output
                                        pageid = newpageid
                            try:
                                output = AnyParseStream(
                                    pageid = f"{pageid}",
                                    metas = metadata,
                                    content = item.content,
                                    elapse_times = item.elapse_times
                                )                            
                            except:
                                output = AnyParseStream(
                                    pageid = f"{pageid}",
                                    metas = metadata,
                                    content = "",
                                    elapse_times={
                                        "datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    }
                                )
                                
                            finally:
                                if len(output.content) > 0:
                                    yield output
                        elapse_time = time.time() - elapse_time
                        elapse_time = {
                            "total": elapse_time
                        }
                        output = AnyParseStream(
                            pageid = f"{pageid}",
                            metas = metadata,
                            content = "",
                            elapse_times = elapse_time
                        )
                        yield output
                    return pdf_parse_stream()
                                
            case ft if ft in self.file_type_instance.docx:
                ### process word
                fileres = self.docxconverter.invoke_docx_v2(
                    processfile,
                    self.ocrmodel,
                    ocr_mode=ocr_mode,
                    **kwargs
                )
                for idx,line in enumerate(fileres):
                    pagecontent = Pagedata(
                        pageid = f"{idx+1}",
                        meta = line['type'],
                        content = line['content'],
                        # images = line['images']
                    )
                    pagescontentlist.append(pagecontent)
                                    
            case ft if ft in self.file_type_instance.pptx:
                ### process ppt
                fileres = self.pptxconverter.invoke_pptx(
                    processfile,
                    self.ocrmodel,
                    ocr_mode=ocr_mode,
                    **kwargs
                )
                for idx,line in enumerate(fileres):
                    pagecontent = Pagedata(
                        pageid = f"{idx+1}",
                        meta = line['type'],
                        content = line['content'],
                        # images = line['images']
                    )
                    pagescontentlist.append(pagecontent)
                    
            case ft if ft in self.file_type_instance.table:
                ### process table
                table_chunk_size = kwargs.pop('table_chunk_size',self.settings.table_chunk_size)
                table_encoding = kwargs.pop('table_encoding',self.settings.table_encoding)
                if ft == "csv":
                    ### process csv
                    if autodetect_encoding:
                        encodings = detect_file_encodings(processfile,file_type="csv")
                        if encodings:
                            table_encoding = encodings
                    # print('table_encoding::: ', table_encoding)
                    fileres = self.tableconverter.invoke_csv(
                        processfile,
                        table_chunk_size,
                        encoding = table_encoding,
                    )
                else:
                    ### process excel
                    fileres = self.tableconverter.invoke_excel(
                        processfile,
                        table_chunk_size
                    )
                for idx,line in enumerate(fileres):
                    sheet_name = line['sheet_name'].strip()
                    sheet_name = f"sheet_name: {sheet_name}\n" if sheet_name else ""
                    pagecontent = Pagedata(
                        pageid = f"{idx+1}",
                        meta = line['type'],
                        content = f"{sheet_name}{line['content'].strip()}"
                    )
                    pagescontentlist.append(pagecontent)
                                   
            case _:
                raise Exception('file type is not allow.')
        document.pages = pagescontentlist
        elapse_time = time.time() - elapse_time
        document.elapse_times = {
            "total": elapse_time
        }
        return document                  
                    