import warnings
warnings.filterwarnings("ignore")
import os
import time
from pathlib import Path
from .base import BaseParser,SyncExecutorMixin,AsyncExecutorMixin
from ..schemas import Page
from ..exceptions import AnyFileNotFoundError,AnyFileTypeError
from ..__version__ import version


class AnyParser(SyncExecutorMixin, BaseParser):
    """
    解析各种文件为markdown
    """
    def __init__(
        self, 
        model_id: str = "", 
        config: dict | str | os.PathLike = {}
    ):
        self._setup(config)
        self._model_id = model_id or self.config.get("model_id", "SyncAnyParse")
        self._version_name = f"{version.strip().lower()}"
    
    def _execute(
        self, 
        file: str | os.PathLike = '', 
        **kwargs
    ):
        processfile = Path(file).expanduser().resolve()
        if not processfile.exists():
            raise AnyFileNotFoundError(f"No such file or directory: {processfile.as_posix()}")
        
        document,metadata = self.build_document(
            file = processfile, 
            **kwargs
        )
        realtime_args,kwargs = self.build_parser_payload(
            file = processfile,
            metadata = metadata,
            **kwargs
        )  
        elapse_time = time.perf_counter()
        if metadata.file_type in self.filetypes_instance.text:
            fileres = self.converters.text.invoke(
                processfile,
                encoding = realtime_args.text_encoding,
                parse_callback = realtime_args.parse_callback,
                **kwargs                        
            )
            for idx,line in enumerate(fileres):
                pagecontent = Page(
                    id = idx+1,
                    content = line['content'],
                    elapse_times = line['time_elapse']
                )
                document.pages.append(pagecontent)
        
        elif metadata.file_type in self.filetypes_instance.markdown:
            fileres = self.converters.markdown.invoke(
                processfile,
                ftype = metadata.file_type,
                encoding = realtime_args.text_encoding,
                parse_callback = realtime_args.parse_callback,
                **kwargs                      
            )
            for idx,line in enumerate(fileres):      
                pagecontent = Page(
                    id = idx+1,
                    content = line['content'],
                    elapse_times = line['time_elapse']
                )
                document.pages.append(pagecontent)      
                             
        elif metadata.file_type in self.filetypes_instance.html:
            fileres = self.converters.html.invoke(
                processfile,
                encoding = realtime_args.text_encoding,
                parse_callback = realtime_args.parse_callback,
                **kwargs                   
            )   
            for idx,line in enumerate(fileres):         
                pagecontent = Page(
                    id = idx+1,
                    content = line['content'],
                    elapse_times = line['time_elapse']
                )
                document.pages.append(pagecontent)    
                      
        elif metadata.file_type in self.filetypes_instance.epub:
            fileres = self.converters.epub.invoke(
                processfile,
                encoding = realtime_args.text_encoding,
                parse_callback = realtime_args.parse_callback,
                **kwargs                   
            )
            for idx,line in enumerate(fileres):         
                pagecontent = Page(
                    id = idx+1,
                    epub_idx = line['epub_idx'],
                    title = line['title'],
                    content = line['content'],
                    elapse_times = line['time_elapse']
                )
                document.pages.append(pagecontent)       
                    
        elif metadata.file_type in self.filetypes_instance.email:
            fileres = self.converters.email.invoke(
                processfile,
                ftype = metadata.file_type,
                encoding = realtime_args.text_encoding,
                parse_callback = realtime_args.parse_callback,
                **kwargs                   
            )
            for idx,line in enumerate(fileres):         
                pagecontent = Page(
                    id = idx+1,
                    title = line['title'],
                    content = line['content'],
                    elapse_times = line['time_elapse']
                )
                document.pages.append(pagecontent)  
                    
        elif metadata.file_type in self.filetypes_instance.ipynb:
            fileres = self.converters.ipynb.invoke(
                processfile,
                encoding = realtime_args.text_encoding,
                parse_callback = realtime_args.parse_callback,
                **kwargs                   
            )
            for idx,line in enumerate(fileres):         
                pagecontent = Page(
                    id = idx+1,
                    cell_idx = line['cell_idx'],
                    title = line['title'],
                    content = line['content'],
                    elapse_times = line['time_elapse']
                )
                document.pages.append(pagecontent)     
                
        elif metadata.file_type in self.filetypes_instance.image:
            fileres = self.converters.pdf.invoke_image(
                processfile,
                ocr_model = self.ocr_model,
                use_doc_cls = realtime_args.use_doc_cls,
                use_doc_rectifier = realtime_args.use_doc_rectifier,
                use_doc_layout = realtime_args.use_doc_layout,
                doc_layout_image_min_size = realtime_args.doc_layout_image_min_size,
                use_image_resize = realtime_args.use_image_resize,
                max_new_tokens = realtime_args.max_new_tokens,
                image_batch_size = realtime_args.image_batch_size,
                ocr_batch_size = realtime_args.ocr_batch_size,
                verbose = realtime_args.verbose,
                parse_callback = realtime_args.parse_callback,
                **kwargs,
            )
            fileres = fileres[0] if fileres else []
            for idx,line in enumerate(fileres):
                pagecontent = Page(
                    id = idx+1,
                    content = line['content'],
                    layout = line['layout'],
                    elapse_times = line['time_elapse']
                )
                document.pages.append(pagecontent)    

        elif metadata.file_type in self.filetypes_instance.pdf:
            if not realtime_args.stream:
                fileres = self.converters.pdf.invoke_pdf(
                    file = processfile,
                    ocr_model = self.ocr_model,
                    use_doc_cls = realtime_args.use_doc_cls,
                    use_doc_rectifier = realtime_args.use_doc_rectifier,
                    use_doc_layout = realtime_args.use_doc_layout,
                    doc_layout_image_min_size = realtime_args.doc_layout_image_min_size,
                    use_image_resize = realtime_args.use_image_resize,
                    max_new_tokens = realtime_args.max_new_tokens,
                    image_batch_size = realtime_args.image_batch_size,
                    ocr_batch_size = realtime_args.ocr_batch_size,
                    dpi = realtime_args.dpi,
                    verbose = realtime_args.verbose,
                    parse_callback = realtime_args.parse_callback,
                    **kwargs,
                )                                        
                for idx,line in enumerate(fileres):
                    pagecontent = Page(
                        id = idx+1,
                        content = line['content'],
                        layout = line['layout'],
                        elapse_times = line['time_elapse']
                    )
                    document.pages.append(pagecontent)    
            else:
                def pdf_parse_stream():
                    fileres = self.converters.pdf.invoke_pdf_stream(
                        file = processfile,
                        ocr_model = self.ocr_model,
                        use_doc_cls = realtime_args.use_doc_cls,
                        use_doc_rectifier = realtime_args.use_doc_rectifier,
                        use_doc_layout = realtime_args.use_doc_layout,
                        doc_layout_image_min_size = realtime_args.doc_layout_image_min_size,
                        use_image_resize = realtime_args.use_image_resize,
                        max_new_tokens = realtime_args.max_new_tokens,
                        image_batch_size = realtime_args.image_batch_size,
                        ocr_batch_size = realtime_args.ocr_batch_size,
                        dpi = realtime_args.dpi,
                        verbose = realtime_args.verbose,
                        parse_callback = realtime_args.parse_callback,
                        **kwargs,
                    )       
                    for idx,line in enumerate(fileres):
                        if line is None:
                            continue
                        pagecontent = Page(
                            id = idx+1,
                            content = line['content'],
                            layout = line['layout'],
                            elapse_times = line['time_elapse']
                        )
                        document.pages = [pagecontent]
                        yield document 
                return pdf_parse_stream()                             
        
        elif metadata.file_type in self.filetypes_instance.office:
            if metadata.file_type in ["docx"]:            
                ### process word
                fileres = self.converters.docx.invoke(
                    processfile,
                    docx_extract_headers_footers = realtime_args.docx_extract_headers_footers,
                    docx_extract_images = realtime_args.docx_extract_images,
                    pdflconverter = self.converters.pdf,
                    ocr_model = self.ocr_model,
                    use_doc_cls = realtime_args.use_doc_cls,
                    use_doc_rectifier = realtime_args.use_doc_rectifier,
                    use_doc_layout = realtime_args.use_doc_layout,
                    doc_layout_image_min_size = realtime_args.doc_layout_image_min_size,
                    use_image_resize = realtime_args.use_image_resize,
                    max_new_tokens = realtime_args.max_new_tokens,
                    image_batch_size = realtime_args.image_batch_size,
                    ocr_batch_size = realtime_args.ocr_batch_size,
                    verbose = realtime_args.verbose,
                    parse_callback = realtime_args.parse_callback,
                    **kwargs
                )
                for idx,line in enumerate(fileres):
                    pagecontent = Page(
                        id = idx+1,
                        content = line['content'],
                        elapse_times = line['time_elapse'],
                        layout = line.get("layout", [])
                    )
                    document.pages.append(pagecontent)
                                    
            elif metadata.file_type in ["pptx"]:
                ### process ppt
                fileres = self.converters.pptx.invoke(
                    processfile,
                    pptx_extract_images = realtime_args.pptx_extract_images,
                    pdflconverter = self.converters.pdf,
                    ocr_model = self.ocr_model,
                    use_doc_cls = realtime_args.use_doc_cls,
                    use_doc_rectifier = realtime_args.use_doc_rectifier,
                    use_doc_layout = realtime_args.use_doc_layout,
                    doc_layout_image_min_size = realtime_args.doc_layout_image_min_size,
                    use_image_resize = realtime_args.use_image_resize,
                    max_new_tokens = realtime_args.max_new_tokens,
                    image_batch_size = realtime_args.image_batch_size,
                    ocr_batch_size = realtime_args.ocr_batch_size,
                    verbose = realtime_args.verbose,
                    parse_callback = realtime_args.parse_callback,
                    **kwargs
                )
                for idx,line in enumerate(fileres):
                    pagecontent = Page(
                        id = idx+1,
                        slide_idx = line["slide_idx"],
                        content = line['content'],
                        elapse_times = line['time_elapse'],
                        layout = line.get("layout", [])
                    )
                    document.pages.append(pagecontent)
            
            elif metadata.file_type in ["xlsx"]:
                ### process excel
                fileres = self.converters.xlsx.invoke(
                    processfile,
                    excel_max_rows = realtime_args.excel_max_rows,
                    excel_extract_images = realtime_args.excel_extract_images,
                    pdflconverter = self.converters.pdf,
                    ocr_model = self.ocr_model,
                    use_doc_cls = realtime_args.use_doc_cls,
                    use_doc_rectifier = realtime_args.use_doc_rectifier,
                    use_doc_layout = realtime_args.use_doc_layout,
                    doc_layout_image_min_size = realtime_args.doc_layout_image_min_size,
                    use_image_resize = realtime_args.use_image_resize,
                    max_new_tokens = realtime_args.max_new_tokens,
                    image_batch_size = realtime_args.image_batch_size,
                    ocr_batch_size = realtime_args.ocr_batch_size,
                    verbose = realtime_args.verbose,
                    parse_callback = realtime_args.parse_callback,
                    **kwargs
                )
                for idx,line in enumerate(fileres):
                    sheet_name = line['sheet_name'].strip()
                    pagecontent = Page(
                        id = idx+1,
                        sheet_idx = line['sheet_idx'],
                        sheet_name = sheet_name,
                        content = f"{line['content']}",
                        elapse_times = line['time_elapse'],
                        layout = line.get("layout", [])
                    )
                    document.pages.append(pagecontent)     
                    
        elif metadata.file_type in self.filetypes_instance.csv:               
            ### process csv
            # print('table_encoding::: ', table_encoding)
            fileres = self.converters.csv.invoke(
                processfile,
                chunk_size = realtime_args.table_chunk_size,
                ftype = metadata.file_type,
                custom_separator = realtime_args.table_custom_separator,
                encoding = realtime_args.text_encoding,
                auto_sniffer = realtime_args.table_sniffer,
                parse_callback = realtime_args.parse_callback,
                **kwargs
            )
            for idx,line in enumerate(fileres):
                sheet_name = line['sheet_name'].strip()
                pagecontent = Page(
                    id = idx+1,
                    sheet_name = sheet_name,
                    chunk_id = line['chunk_id'],
                    chunk_size = line['chunk_size'],
                    content = f"{line['content']}",
                    elapse_times = line['time_elapse']
                )
                document.pages.append(pagecontent)
        
        else:
            raise AnyFileTypeError(f"file type is not allow. file_type: {metadata.file_type}")
            
        document.content = "\n".join([x.content for x in document.pages])
        elapse_time = time.perf_counter() - elapse_time
        document.elapse_times = elapse_time
        return document         
    
    
class AsyncAnyParser(AsyncExecutorMixin, BaseParser):
    """
    解析各种文件为markdown
    """
    def __init__(
        self, 
        model_id: str = "", 
        config: dict | str | os.PathLike = {}
    ):
        self._setup(config)
        self._model_id = model_id or self.config.get("model_id", "AsyncAnyParse")
        self._version_name = f"{version.strip().lower()}"
    
    async def _aexecute(
        self, 
        file: str | os.PathLike = '', 
        **kwargs
    ):
        processfile = Path(file).expanduser().resolve()
        if not processfile.exists():
            raise AnyFileNotFoundError(f"No such file or directory: {processfile.as_posix()}")
        
        document,metadata = self.build_document(
            file = processfile, 
            **kwargs
        )
        realtime_args,kwargs = self.build_parser_payload(
            file = processfile,
            metadata = metadata,
            **kwargs
        )
        elapse_time = time.perf_counter()
        if metadata.file_type in self.filetypes_instance.text:
            fileres = await self.converters.text.ainvoke(
                processfile,
                encoding = realtime_args.text_encoding,
                parse_callback = realtime_args.parse_callback,
                **kwargs                        
            )
            for idx,line in enumerate(fileres):
                pagecontent = Page(
                    id = idx+1,
                    content = line['content'],
                    elapse_times = line['time_elapse']
                )
                document.pages.append(pagecontent)
        
        elif metadata.file_type in self.filetypes_instance.markdown:
            fileres = await self.converters.markdown.ainvoke(
                processfile,
                ftype = metadata.file_type,
                encoding = realtime_args.text_encoding,
                parse_callback = realtime_args.parse_callback,
                **kwargs                      
            )
            for idx,line in enumerate(fileres):      
                pagecontent = Page(
                    id = idx+1,
                    content = line['content'],
                    elapse_times = line['time_elapse']
                )
                document.pages.append(pagecontent)      
                             
        elif metadata.file_type in self.filetypes_instance.html:
            fileres = await self.converters.html.ainvoke(
                processfile,
                encoding = realtime_args.text_encoding,
                parse_callback = realtime_args.parse_callback,
                **kwargs                   
            )   
            for idx,line in enumerate(fileres):         
                pagecontent = Page(
                    id = idx+1,
                    content = line['content'],
                    elapse_times = line['time_elapse']
                )
                document.pages.append(pagecontent)    
                      
        elif metadata.file_type in self.filetypes_instance.epub:
            fileres = await self.converters.epub.ainvoke(
                processfile,
                encoding = realtime_args.text_encoding,
                parse_callback = realtime_args.parse_callback,
                **kwargs                   
            )
            for idx,line in enumerate(fileres):         
                pagecontent = Page(
                    id = idx+1,
                    epub_idx = line['epub_idx'],
                    title = line['title'],
                    content = line['content'],
                    elapse_times = line['time_elapse']
                )
                document.pages.append(pagecontent)       
                    
        elif metadata.file_type in self.filetypes_instance.email:
            fileres = await self.converters.email.ainvoke(
                processfile,
                ftype = metadata.file_type,
                encoding = realtime_args.text_encoding,
                parse_callback = realtime_args.parse_callback,
                **kwargs                   
            )
            for idx,line in enumerate(fileres):         
                pagecontent = Page(
                    id = idx+1,
                    title = line['title'],
                    content = line['content'],
                    elapse_times = line['time_elapse']
                )
                document.pages.append(pagecontent)  
                    
        elif metadata.file_type in self.filetypes_instance.ipynb:
            fileres = await self.converters.ipynb.ainvoke(
                processfile,
                encoding = realtime_args.text_encoding,
                parse_callback = realtime_args.parse_callback,
                **kwargs                   
            )
            for idx,line in enumerate(fileres):         
                pagecontent = Page(
                    id = idx+1,
                    cell_idx = line['cell_idx'],
                    title = line['title'],
                    content = line['content'],
                    elapse_times = line['time_elapse']
                )
                document.pages.append(pagecontent)     
                
        elif metadata.file_type in self.filetypes_instance.image:
            fileres = await self.converters.pdf.ainvoke_image(
                processfile,
                ocr_model = self.ocr_model,
                use_doc_cls = realtime_args.use_doc_cls,
                use_doc_rectifier = realtime_args.use_doc_rectifier,
                use_doc_layout = realtime_args.use_doc_layout,
                doc_layout_image_min_size = realtime_args.doc_layout_image_min_size,
                use_image_resize = realtime_args.use_image_resize,
                max_new_tokens = realtime_args.max_new_tokens,
                image_batch_size = realtime_args.image_batch_size,
                ocr_batch_size = realtime_args.ocr_batch_size,
                verbose = realtime_args.verbose,
                parse_callback = realtime_args.parse_callback,
                **kwargs,
            )
            fileres = fileres[0] if fileres else []
            for idx,line in enumerate(fileres):
                pagecontent = Page(
                    id = idx+1,
                    content = line['content'],
                    layout = line['layout'],
                    elapse_times = line['time_elapse']
                )
                document.pages.append(pagecontent)    

        elif metadata.file_type in self.filetypes_instance.pdf:
            if not realtime_args.stream:
                fileres = await self.converters.pdf.ainvoke_pdf(
                    file = processfile,
                    ocr_model = self.ocr_model,
                    use_doc_cls = realtime_args.use_doc_cls,
                    use_doc_rectifier = realtime_args.use_doc_rectifier,
                    use_doc_layout = realtime_args.use_doc_layout,
                    doc_layout_image_min_size = realtime_args.doc_layout_image_min_size,
                    use_image_resize = realtime_args.use_image_resize,
                    max_new_tokens = realtime_args.max_new_tokens,
                    image_batch_size = realtime_args.image_batch_size,
                    ocr_batch_size = realtime_args.ocr_batch_size,
                    dpi = realtime_args.dpi,
                    verbose = realtime_args.verbose,
                    parse_callback = realtime_args.parse_callback,
                    **kwargs,
                )                                        
                for idx,line in enumerate(fileres):
                    pagecontent = Page(
                        id = idx+1,
                        content = line['content'],
                        layout = line['layout'],
                        elapse_times = line['time_elapse']
                    )
                    document.pages.append(pagecontent)    
            else:
                async def apdf_parse_stream():
                    fileres = self.converters.pdf.ainvoke_pdf_stream(
                        file = processfile,
                        ocr_model = self.ocr_model,
                        use_doc_cls = realtime_args.use_doc_cls,
                        use_doc_rectifier = realtime_args.use_doc_rectifier,
                        use_doc_layout = realtime_args.use_doc_layout,
                        doc_layout_image_min_size = realtime_args.doc_layout_image_min_size,
                        use_image_resize = realtime_args.use_image_resize,
                        max_new_tokens = realtime_args.max_new_tokens,
                        image_batch_size = realtime_args.image_batch_size,
                        ocr_batch_size = realtime_args.ocr_batch_size,
                        dpi = realtime_args.dpi,
                        verbose = realtime_args.verbose,
                        parse_callback = realtime_args.parse_callback,
                        **kwargs,
                    )       
                    idx = 0
                    async for line in fileres:
                        # print("line::: ", line)
                        idx += 1
                        if line is None:
                            continue
                        pagecontent = Page(
                            id = idx,
                            content = line['content'],
                            layout = line['layout'],
                            elapse_times = line['time_elapse']
                        )           
                        document.pages = [pagecontent]
                        yield document 
                return apdf_parse_stream()                             
        
        elif metadata.file_type in self.filetypes_instance.office:
            if metadata.file_type in ["docx"]:            
                ### process word
                fileres = await self.converters.docx.ainvoke(
                    processfile,
                    docx_extract_headers_footers = realtime_args.docx_extract_headers_footers,
                    docx_extract_images = realtime_args.docx_extract_images,
                    pdflconverter = self.converters.pdf,
                    ocr_model = self.ocr_model,
                    use_doc_cls = realtime_args.use_doc_cls,
                    use_doc_rectifier = realtime_args.use_doc_rectifier,
                    use_doc_layout = realtime_args.use_doc_layout,
                    doc_layout_image_min_size = realtime_args.doc_layout_image_min_size,
                    use_image_resize = realtime_args.use_image_resize,
                    max_new_tokens = realtime_args.max_new_tokens,
                    image_batch_size = realtime_args.image_batch_size,
                    ocr_batch_size = realtime_args.ocr_batch_size,
                    verbose = realtime_args.verbose,
                    parse_callback = realtime_args.parse_callback,
                    **kwargs
                )
                for idx,line in enumerate(fileres):
                    pagecontent = Page(
                        id = idx+1,
                        content = line['content'],
                        elapse_times = line['time_elapse'],
                        layout = line.get("layout", [])
                    )
                    document.pages.append(pagecontent)
                                    
            elif metadata.file_type in ["pptx"]:
                ### process ppt
                fileres = await self.converters.pptx.ainvoke(
                    processfile,
                    pptx_extract_images = realtime_args.pptx_extract_images,
                    pdflconverter = self.converters.pdf,
                    ocr_model = self.ocr_model,
                    use_doc_cls = realtime_args.use_doc_cls,
                    use_doc_rectifier = realtime_args.use_doc_rectifier,
                    use_doc_layout = realtime_args.use_doc_layout,
                    doc_layout_image_min_size = realtime_args.doc_layout_image_min_size,
                    use_image_resize = realtime_args.use_image_resize,
                    max_new_tokens = realtime_args.max_new_tokens,
                    image_batch_size = realtime_args.image_batch_size,
                    ocr_batch_size = realtime_args.ocr_batch_size,
                    verbose = realtime_args.verbose,
                    parse_callback = realtime_args.parse_callback,
                    **kwargs
                )
                for idx,line in enumerate(fileres):
                    pagecontent = Page(
                        id = idx+1,
                        slide_idx = line["slide_idx"],
                        content = line['content'],
                        elapse_times = line['time_elapse'],
                        layout = line.get("layout", [])
                    )
                    document.pages.append(pagecontent)
            
            elif metadata.file_type in ["xlsx"]:
                ### process excel
                fileres = await self.converters.xlsx.ainvoke(
                    processfile,
                    excel_max_rows = realtime_args.excel_max_rows,
                    excel_extract_images = realtime_args.excel_extract_images,
                    pdflconverter = self.converters.pdf,
                    ocr_model = self.ocr_model,
                    use_doc_cls = realtime_args.use_doc_cls,
                    use_doc_rectifier = realtime_args.use_doc_rectifier,
                    use_doc_layout = realtime_args.use_doc_layout,
                    doc_layout_image_min_size = realtime_args.doc_layout_image_min_size,
                    use_image_resize = realtime_args.use_image_resize,
                    max_new_tokens = realtime_args.max_new_tokens,
                    image_batch_size = realtime_args.image_batch_size,
                    ocr_batch_size = realtime_args.ocr_batch_size,
                    verbose = realtime_args.verbose,
                    parse_callback = realtime_args.parse_callback,
                    **kwargs
                )
                for idx,line in enumerate(fileres):
                    sheet_name = line['sheet_name'].strip()
                    pagecontent = Page(
                        id = idx+1,
                        sheet_idx = line['sheet_idx'],
                        sheet_name = sheet_name,
                        content = f"{line['content']}",
                        elapse_times = line['time_elapse'],
                        layout = line.get("layout", [])
                    )
                    document.pages.append(pagecontent)     
                    
        elif metadata.file_type in self.filetypes_instance.csv:               
            ### process csv
            # print('table_encoding::: ', table_encoding)
            fileres = await self.converters.csv.ainvoke(
                processfile,
                chunk_size = realtime_args.table_chunk_size,
                ftype = metadata.file_type,
                custom_separator = realtime_args.table_custom_separator,
                encoding = realtime_args.text_encoding,
                auto_sniffer = realtime_args.table_sniffer,
                parse_callback = realtime_args.parse_callback,
                **kwargs
            )
            for idx,line in enumerate(fileres):
                sheet_name = line['sheet_name'].strip()
                pagecontent = Page(
                    id = idx+1,
                    sheet_name = sheet_name,
                    chunk_id = line['chunk_id'],
                    chunk_size = line['chunk_size'],
                    content = f"{line['content']}",
                    elapse_times = line['time_elapse']
                )
                document.pages.append(pagecontent)
        
        else:
            raise AnyFileTypeError(f"file type is not allow. file_type: {metadata.file_type}")
            
        document.content = "\n".join([x.content for x in document.pages])
        elapse_time = time.perf_counter() - elapse_time
        document.elapse_times = elapse_time
        return document         
    
             