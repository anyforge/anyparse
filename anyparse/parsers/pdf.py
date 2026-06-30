import os
import re
import time
import threading
import traceback
import pypdfium2 as pdfium
from PIL import Image
from rich.progress import track
from typing import List,BinaryIO
from ..schemas import Element,AnyOCROutput
from ..loaders import autoload_image
from ..utils.pdfcls import PdfClassifier
from ..exceptions import *
from .base import BaseConverter


class PdfConverter(BaseConverter):
    _pdfium_lock = threading.RLock()
    def __init__(self, **kwargs):
        self.kwargs = kwargs 
        self.pdfclf = PdfClassifier(**kwargs)
        
    def format_text(self, result_str: str, label_name: str):
        result_str = re.sub(r'\n{2,}', '\n', result_str).strip()
        return result_str
    
    def invoke_item(
        self, 
        images: List[Image.Image], 
        ocr_model, 
        use_doc_cls: bool = False,
        use_doc_rectifier: bool = False,
        use_doc_layout: bool = True,
        doc_layout_image_min_size: int = 400,
        use_image_resize: bool = True,
        max_new_tokens: int = 16384,
        image_batch_size: int = None,
        ocr_batch_size: int = None,
        **kwargs
    ):
        try:
            output = []
            ocr_output: List[AnyOCROutput] = ocr_model.invoke(
                images = images, 
                image_batch_size = image_batch_size,
                ocr_batch_size = ocr_batch_size,
                use_doc_cls = use_doc_cls,
                use_doc_rectifier = use_doc_rectifier,
                use_doc_layout = use_doc_layout,
                doc_layout_image_min_size = doc_layout_image_min_size,
                use_image_resize = use_image_resize,
                max_new_tokens = max_new_tokens,
                **kwargs                
            )
            for item in ocr_output:
                page_output = []
                for block in item.blocks:
                    block: Element = block
                    page_output.append(block.model_dump())
                output.append(page_output)
        except:       
            traceback.print_exc() 
            output = []
        finally:
            return output   
    
    async def ainvoke_item(
        self, 
        images: List[Image.Image], 
        ocr_model, 
        use_doc_cls: bool = False,
        use_doc_rectifier: bool = False,
        use_doc_layout: bool = True,
        doc_layout_image_min_size: int = 400,
        use_image_resize: bool = True,
        max_new_tokens: int = 16384,
        image_batch_size: int = None,
        ocr_batch_size: int = None,
        **kwargs
    ):
        try:
            output = []
            ocr_output: List[AnyOCROutput] = await ocr_model.ainvoke(
                images = images, 
                image_batch_size = image_batch_size,
                ocr_batch_size = ocr_batch_size,
                use_doc_cls = use_doc_cls,
                use_doc_rectifier = use_doc_rectifier,
                use_doc_layout = use_doc_layout,
                doc_layout_image_min_size = doc_layout_image_min_size,
                use_image_resize = use_image_resize,
                max_new_tokens = max_new_tokens,
                **kwargs                
            )
            # print("ocr_output::: ", ocr_output)
            for item in ocr_output:
                page_output = []
                for block in item.blocks:
                    block: Element = block
                    page_output.append(block.model_dump())
                output.append(page_output)
        except:       
            traceback.print_exc() 
            output = []
        finally:
            return output      
         
    def invoke_image(
        self, 
        file: str 
            | os.PathLike 
            | Image.Image 
            | BinaryIO 
            | bytes 
            | List[str] 
            | List[os.PathLike] 
            | List[Image.Image] 
            | List[BinaryIO] 
            | List[bytes], 
        ocr_model, 
        use_doc_cls: bool = False,
        use_doc_rectifier: bool = False,
        use_doc_layout: bool = True,
        doc_layout_image_min_size: int = 400,
        use_image_resize: bool = True,
        max_new_tokens: int = 16384,
        image_batch_size: int = None,
        ocr_batch_size: int = None,
        verbose: bool = True,
        **kwargs
    ) -> List[dict]:
        try:
            parse_callback = kwargs.get("parse_callback")
            if not isinstance(file, list):
                file = [file]
                
            res = []
            for i in range(0, len(file), image_batch_size):
                try:
                    batch_file = file[i:i+image_batch_size]
                    start_time = time.perf_counter()      
                    batch_images = autoload_image(batch_file)
                    parse_callback.on_started(**{
                        "file": batch_images
                    })                            
                    page_outputs = self.invoke_item(
                        images = batch_images, 
                        ocr_model = ocr_model,
                        use_doc_cls = use_doc_cls,
                        use_doc_rectifier = use_doc_rectifier,
                        use_doc_layout = use_doc_layout,
                        doc_layout_image_min_size = doc_layout_image_min_size,
                        use_image_resize = use_image_resize,
                        max_new_tokens = max_new_tokens,
                        image_batch_size = image_batch_size,
                        ocr_batch_size = ocr_batch_size,
                        verbose = verbose,
                        **kwargs
                    )
                    parse_callback.on_finished(**{
                        "file": batch_images
                    })   
                except:
                    traceback.print_exc()
                    page_outputs = [[]] * len(batch_images)               
                time_res = time.perf_counter() - start_time
                time_res = time_res / (len(page_outputs) or 1)
                output = []
                for page_output in page_outputs:
                    content = [self.format_text(x['content'],x['label']) for x in page_output]
                    content = [x for x in content if x.strip()]
                    content = '\n'.join(content)
                    output.append({
                        "type": "image",
                        "layout": page_output,
                        "content": content,
                        "time_elapse": time_res
                    })                    
                # parse_callback.on_finished(**{
                #     "file": batch_file
                # })   
                res.append(output)
        except:       
            traceback.print_exc() 
            res = []
        finally:
            return res      
        
    async def ainvoke_image(
        self, 
        file: str 
            | os.PathLike 
            | Image.Image 
            | BinaryIO 
            | bytes 
            | List[str] 
            | List[os.PathLike] 
            | List[Image.Image] 
            | List[BinaryIO] 
            | List[bytes], 
        ocr_model, 
        use_doc_cls: bool = False,
        use_doc_rectifier: bool = False,
        use_doc_layout: bool = True,
        doc_layout_image_min_size: int = 400,
        use_image_resize: bool = True,
        max_new_tokens: int = 16384,
        image_batch_size: int = None,
        ocr_batch_size: int = None,
        verbose: bool = True,
        **kwargs
    ) -> List[dict]:
        try:
            parse_callback = kwargs.get("parse_callback")
            if not isinstance(file, list):
                file = [file]
                
            res = []
            for i in range(0, len(file), image_batch_size):
                try:
                    batch_file = file[i:i+image_batch_size]
                    start_time = time.perf_counter()      
                    batch_images = autoload_image(batch_file)
                    parse_callback.on_started(**{
                        "file": batch_images
                    })                            
                    page_outputs = await self.ainvoke_item(
                        images = batch_images, 
                        ocr_model = ocr_model,
                        use_doc_cls = use_doc_cls,
                        use_doc_rectifier = use_doc_rectifier,
                        use_doc_layout = use_doc_layout,
                        doc_layout_image_min_size = doc_layout_image_min_size,
                        use_image_resize = use_image_resize,
                        max_new_tokens = max_new_tokens,
                        image_batch_size = image_batch_size,
                        ocr_batch_size = ocr_batch_size,
                        verbose = verbose,
                        **kwargs
                    )
                    parse_callback.on_finished(**{
                        "file": batch_images
                    })   
                except:
                    traceback.print_exc()
                    page_outputs = [[]] * len(batch_images)               
                time_res = time.perf_counter() - start_time
                time_res = time_res / (len(page_outputs) or 1)
                output = []
                for page_output in page_outputs:
                    content = [self.format_text(x['content'],x['label']) for x in page_output]
                    content = [x for x in content if x.strip()]
                    content = '\n'.join(content)
                    output.append({
                        "type": "image",
                        "layout": page_output,
                        "content": content,
                        "time_elapse": time_res
                    })                    
                # parse_callback.on_finished(**{
                #     "file": batch_file
                # })   
                res.append(output)
        except:       
            traceback.print_exc() 
            res = []
        finally:
            return res      
        
    def render_page_to_image(
        self,
        page: pdfium.PdfPage,
        dpi: int = 200,
        draw_annots: bool = False # 是否绘制注释
    ) -> Image:
        scale = dpi / 72
        image = page.render(
            scale = scale,
            draw_annots=draw_annots
        ).to_pil()
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image        
        
    def invoke_pdf(
        self, 
        file: str | os.PathLike,  
        ocr_model, 
        use_doc_cls: bool = False,
        use_doc_rectifier: bool = False,
        use_doc_layout: bool = True,
        doc_layout_image_min_size: int = 400,
        use_image_resize: bool = True,
        max_new_tokens: int = 16384,
        dpi: int = 200,
        draw_annots: bool = False,
        image_batch_size: int = None,
        ocr_batch_size: int = None,
        verbose: bool = True,
        **kwargs
    ):
        pdfdoc = None
        try:
            parse_callback = kwargs.get("parse_callback")
            parse_callback.on_started(**{
                "file": file
            })
            res = []
            with self._pdfium_lock:  #  加锁保护
                pdfdoc = pdfium.PdfDocument(file.as_posix())
            total_page_count = len(pdfdoc)
            # 处理每一页
            total_range = list(range(total_page_count))
            batch_range = list(range(0, len(total_range), image_batch_size))
            batch_range = track(
                batch_range, 
                total = len(batch_range), 
                description=f"File[{file.name}]:",
                disable=not verbose,
                refresh_per_second=1
            )
            for i in batch_range:
                start_time = time.perf_counter()
                batch_range = total_range[i:i+image_batch_size]
                batch_images = []
                try:
                    for page_number in batch_range:
                        page = pdfdoc[page_number]
                        image = self.render_page_to_image(
                            page = page,
                            dpi = dpi,
                            draw_annots = draw_annots
                        )
                        batch_images.append(image)

                    page_outputs = self.invoke_item(
                        images = batch_images, 
                        ocr_model = ocr_model,
                        use_doc_cls = use_doc_cls,
                        use_doc_rectifier = use_doc_rectifier,
                        use_doc_layout = use_doc_layout,
                        doc_layout_image_min_size = doc_layout_image_min_size,
                        use_image_resize = use_image_resize,
                        max_new_tokens = max_new_tokens,
                        image_batch_size = image_batch_size,
                        ocr_batch_size = ocr_batch_size,
                        verbose = verbose,
                        **kwargs
                    ) 
                except:
                    traceback.print_exc()
                    page_outputs = [[]] * len(batch_images)
                time_res = time.perf_counter() - start_time
                time_res = time_res / (len(page_outputs) or 1)
                for page_number,page_output in zip(batch_range,page_outputs):                
                    content = [self.format_text(x['content'],x['label']) for x in page_output]
                    content = [x for x in content if x.strip()]
                    content = '\n'.join(content)            
                    page_res = {
                        "pageid": page_number + 1,
                        "type": "pdf",
                        "layout": page_output,
                        "content": content,
                        "time_elapse": time_res
                    }
                    res.append(page_res)            
            parse_callback.on_finished(**{
                "file": file
            })  
        except:
            traceback.print_exc()
            res = []
        finally:
            # 3. 强制清理：这是最关键的一步
            if pdfdoc is not None:
                with self._pdfium_lock:
                    try:
                        pdfdoc.close()  #  显式关闭
                    except:
                        pass  # 忽略关闭时的异常
            return res
        
    async def ainvoke_pdf(
        self, 
        file: str | os.PathLike,  
        ocr_model, 
        use_doc_cls: bool = False,
        use_doc_rectifier: bool = False,
        use_doc_layout: bool = True,
        doc_layout_image_min_size: int = 400,
        use_image_resize: bool = True,
        max_new_tokens: int = 16384,
        dpi: int = 200,
        draw_annots: bool = False,
        image_batch_size: int = None,
        ocr_batch_size: int = None,
        verbose: bool = True,
        **kwargs
    ):
        pdfdoc = None
        try:
            parse_callback = kwargs.get("parse_callback")
            parse_callback.on_started(**{
                "file": file
            })
            res = []
            with self._pdfium_lock:  #  加锁保护
                pdfdoc = pdfium.PdfDocument(file.as_posix())
            total_page_count = len(pdfdoc)
            # 处理每一页
            total_range = list(range(total_page_count))
            batch_range = list(range(0, len(total_range), image_batch_size))
            batch_range = track(
                batch_range, 
                total = len(batch_range), 
                description=f"File[{file.name}]:",
                disable=not verbose,
                refresh_per_second=1
            )
            for i in batch_range:
                start_time = time.perf_counter()
                batch_range = total_range[i:i+image_batch_size]
                batch_images = []
                try:
                    for page_number in batch_range:
                        page = pdfdoc[page_number]
                        image = self.render_page_to_image(
                            page = page,
                            dpi = dpi,
                            draw_annots = draw_annots
                        )
                        batch_images.append(image)

                    page_outputs = await self.ainvoke_item(
                        images = batch_images, 
                        ocr_model = ocr_model,
                        use_doc_cls = use_doc_cls,
                        use_doc_rectifier = use_doc_rectifier,
                        use_doc_layout = use_doc_layout,
                        doc_layout_image_min_size = doc_layout_image_min_size,
                        use_image_resize = use_image_resize,
                        max_new_tokens = max_new_tokens,
                        image_batch_size = image_batch_size,
                        ocr_batch_size = ocr_batch_size,
                        verbose = verbose,
                        **kwargs
                    ) 
                except:
                    traceback.print_exc()
                    page_outputs = [[]] * len(batch_images)
                time_res = time.perf_counter() - start_time
                time_res = time_res / (len(page_outputs) or 1)
                for page_number,page_output in zip(batch_range,page_outputs):                
                    content = [self.format_text(x['content'],x['label']) for x in page_output]
                    content = [x for x in content if x.strip()]
                    content = '\n'.join(content)            
                    page_res = {
                        "pageid": page_number + 1,
                        "type": "pdf",
                        "layout": page_output,
                        "content": content,
                        "time_elapse": time_res
                    }
                    res.append(page_res)            
            parse_callback.on_finished(**{
                "file": file
            })  
        except:
            traceback.print_exc()
            res = []
        finally:
            # 3. 强制清理：这是最关键的一步
            if pdfdoc is not None:
                with self._pdfium_lock:
                    try:
                        pdfdoc.close()  #  显式关闭
                    except:
                        pass  # 忽略关闭时的异常
            return res
    
    def invoke_pdf_stream(
        self, 
        file: str | os.PathLike,  
        ocr_model, 
        use_doc_cls: bool = False,
        use_doc_rectifier: bool = False,
        use_doc_layout: bool = True,
        doc_layout_image_min_size: int = 400,
        use_image_resize: bool = True,
        max_new_tokens: int = 16384,
        dpi: int = 200,
        draw_annots: bool = False,
        image_batch_size: int = None,
        ocr_batch_size: int = None,
        verbose: bool = True,
        **kwargs
    ):
        pdfdoc = None
        parse_callback = kwargs.get("parse_callback")
        parse_callback.on_started(**{
            "file": file
        })
        res = []
        with self._pdfium_lock:  #  加锁保护
            pdfdoc = pdfium.PdfDocument(file.as_posix())
        total_page_count = len(pdfdoc)
        # 处理每一页
        total_range = list(range(total_page_count))
        batch_range = list(range(0, len(total_range), image_batch_size))
        batch_range = track(
            batch_range, 
            total = len(batch_range), 
            description=f"File[{file.name}]:",
            disable=not verbose,
            refresh_per_second=1
        )
        for i in batch_range:
            start_time = time.perf_counter()
            batch_range = total_range[i:i+image_batch_size]
            batch_images = []
            try:
                for page_number in batch_range:
                    page = pdfdoc[page_number]
                    image = self.render_page_to_image(
                        page = page,
                        dpi = dpi,
                        draw_annots = draw_annots
                    )
                    batch_images.append(image)

                page_outputs = self.invoke_item(
                    images = batch_images, 
                    ocr_model = ocr_model,
                    use_doc_cls = use_doc_cls,
                    use_doc_rectifier = use_doc_rectifier,
                    use_doc_layout = use_doc_layout,
                    doc_layout_image_min_size = doc_layout_image_min_size,
                    use_image_resize = use_image_resize,
                    max_new_tokens = max_new_tokens,
                    image_batch_size = image_batch_size,
                    ocr_batch_size = ocr_batch_size,
                    verbose = verbose,
                    **kwargs
                ) 
            except:
                traceback.print_exc()
                page_outputs = [[]] * len(batch_range)
            time_res = time.perf_counter() - start_time
            time_res = time_res / len(page_outputs)
            for page_number,page_output in zip(batch_range,page_outputs):                
                content = [self.format_text(x['content'],x['label']) for x in page_output]
                content = [x for x in content if x.strip()]
                content = '\n'.join(content)            
                page_res = {
                    "pageid": page_number + 1,
                    "type": "pdf",
                    "layout": page_output,
                    "content": content,
                    "time_elapse": time_res
                }        
                yield page_res
        
        parse_callback.on_finished(**{
            "file": file
        })  
        # 3. 强制清理：这是最关键的一步
        if pdfdoc is not None:
            with self._pdfium_lock:
                try:
                    pdfdoc.close()  #  显式关闭
                except:
                    pass  # 忽略关闭时的异常
        yield None
        
    async def ainvoke_pdf_stream(
        self, 
        file: str | os.PathLike,  
        ocr_model, 
        use_doc_cls: bool = False,
        use_doc_rectifier: bool = False,
        use_doc_layout: bool = True,
        doc_layout_image_min_size: int = 400,
        use_image_resize: bool = True,
        max_new_tokens: int = 16384,
        dpi: int = 200,
        draw_annots: bool = False,
        image_batch_size: int = None,
        ocr_batch_size: int = None,
        verbose: bool = True,
        **kwargs
    ):
        pdfdoc = None
        parse_callback = kwargs.get("parse_callback")
        parse_callback.on_started(**{
            "file": file
        })
        res = []
        with self._pdfium_lock:  #  加锁保护
            pdfdoc = pdfium.PdfDocument(file.as_posix())
        total_page_count = len(pdfdoc)
        # 处理每一页
        total_range = list(range(total_page_count))
        batch_range = list(range(0, len(total_range), image_batch_size))
        batch_range = track(
            batch_range, 
            total = len(batch_range), 
            description=f"File[{file.name}]:",
            disable=not verbose,
            refresh_per_second=1
        )
        for i in batch_range:
            start_time = time.perf_counter()
            batch_range = total_range[i:i+image_batch_size]
            batch_images = []
            try:
                for page_number in batch_range:
                    page = pdfdoc[page_number]
                    image = self.render_page_to_image(
                        page = page,
                        dpi = dpi,
                        draw_annots = draw_annots
                    )
                    batch_images.append(image)

                page_outputs = await self.ainvoke_item(
                    images = batch_images, 
                    ocr_model = ocr_model,
                    use_doc_cls = use_doc_cls,
                    use_doc_rectifier = use_doc_rectifier,
                    use_doc_layout = use_doc_layout,
                    doc_layout_image_min_size = doc_layout_image_min_size,
                    use_image_resize = use_image_resize,
                    max_new_tokens = max_new_tokens,
                    image_batch_size = image_batch_size,
                    ocr_batch_size = ocr_batch_size,
                    verbose = verbose,
                    **kwargs
                ) 
            except:
                traceback.print_exc()
                page_outputs = [[]] * len(batch_range)
            time_res = time.perf_counter() - start_time
            time_res = time_res / len(page_outputs)
            # print("page_outputs::: ", page_outputs)
            for page_number,page_output in zip(batch_range,page_outputs):                
                content = [self.format_text(x['content'],x['label']) for x in page_output]
                content = [x for x in content if x.strip()]
                content = '\n'.join(content)            
                page_res = {
                    "pageid": page_number + 1,
                    "type": "pdf",
                    "layout": page_output,
                    "content": content,
                    "time_elapse": time_res
                }        
                yield page_res
        
        parse_callback.on_finished(**{
            "file": file
        })  
        # 3. 强制清理：这是最关键的一步
        if pdfdoc is not None:
            with self._pdfium_lock:
                try:
                    pdfdoc.close()  #  显式关闭
                except:
                    pass  # 忽略关闭时的异常
        yield None
        