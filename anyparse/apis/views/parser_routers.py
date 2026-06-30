import json
import uuid
import traceback
import time
import datetime
import shutil
import aiofiles
from pathlib import Path
from typing import Any
from fastapi import Request,APIRouter,Form,UploadFile
from sse_starlette import EventSourceResponse
from .contexts import (
    mainviews, 
    get_appviews_config,
    get_cache_dir,
    default_maxsize,
    default_file_read_size,
)
from ..schemas.responses import anystatus,anyresponse
from ..schemas.openai import FileObject
from ...loggers import logger


def create_parser_routers(public_indexbp: APIRouter, protect_indexbp: APIRouter):    

    api_config = get_appviews_config()
    cache_dir = get_cache_dir()
    @protect_indexbp.get(
        path = api_config.get("modelapi",{}).get("filetypes","/filetypes/v1"),
        summary = "获取文件类型"
    )        
    async def filetypes_apis(
        *,
        request: Request, 
        appviews: Any = mainviews,
    ):    
        argsTime = 0.0
        errorInfo = ""
        rawTime = time.perf_counter()
        callinfo = appviews._call_request_info(request)
        try: 
            ### 检查文件类型
            allow_filetypes = {}
            for x,y in appviews.parser_model.filetypes_instance.model_dump().items():
                allow_filetypes[x] = list(y)

            output = anyresponse.success(
                key = anystatus.success.key,
                code = anystatus.success.code,
                data = allow_filetypes,
                msg = anystatus.success.msg                   
            )   
            return output     
            
        except Exception as e:
            errorInfo = traceback.format_exc()
            output = anyresponse.fail(
                key = anystatus.internal_fail.key,
                code = anystatus.internal_fail.code,
                data = "",
                msg = anystatus.internal_fail.msg              
            )
            return output
        
        finally:
            processTime = time.perf_counter() - rawTime
            appviews._call_api_log(
                callinfo = callinfo,
                argsTime = argsTime,
                processTime = processTime,
                errorInfo = errorInfo,
            )
                
    @protect_indexbp.post(
        path = api_config.get("modelapi",{}).get("invoke", "/invoke/v1"),
        summary="Invoke parser",
    )
    async def parser_apis(
        file: UploadFile = None,
        use_doc_cls: bool = Form(False),
        use_doc_rectifier: bool = Form(False),
        use_doc_layout: bool = Form(True),
        doc_layout_image_min_size: int = Form(500),
        dpi: int = Form(200),
        text_encoding: str = Form("utf-8"),
        table_chunk_size: int = Form(4096),
        table_custom_separator: str = Form(None),
        maxsize: float = Form(default_maxsize),
        file_read_size: int = Form(default_file_read_size),
        verbose: bool = Form(True),
        stream: bool = Form(False),
        autocal_md5: bool = Form(True),
        autodetect_encoding: bool = Form(True),
        docx_extract_headers_footers: bool = Form(True),
        docx_extract_images: bool = Form(True),
        pptx_extract_images: bool = Form(True),
        excel_extract_images: bool = Form(True),
        excel_max_rows: int = Form(None),
        use_image_resize: bool = Form(True),
        max_new_tokens: int = Form(16384),
        image_batch_size: int = Form(1),
        ocr_batch_size: int = Form(1),
        *,
        request: Request, 
        appviews: Any = mainviews,
    ):    
        argsTime = 0.0
        errorInfo = ""
        rawTime = time.perf_counter()
        callinfo = appviews._call_request_info(request)
        try: 
            file_name,file_size,file_suffix = file.filename,file.size,Path(file.filename).suffix
            callin_args = {
                "file_name": file_name,
                "file_size": file_size,
                "file_suffix": file_suffix,
                "use_doc_cls": use_doc_cls,
                "use_doc_rectifier": use_doc_rectifier,
                "use_doc_layout": use_doc_layout,
                "doc_layout_image_min_size": doc_layout_image_min_size,
                "dpi": dpi,
                "text_encoding": text_encoding,
                "table_chunk_size": table_chunk_size,
                "table_custom_separator": table_custom_separator,
                "maxsize": maxsize,
                "file_read_size": file_read_size,
                "verbose": verbose,
                "stream": stream,
                "autocal_md5": autocal_md5,
                "autodetect_encoding": autodetect_encoding,
                "docx_extract_headers_footers": docx_extract_headers_footers,
                "docx_extract_images": docx_extract_images,
                "pptx_extract_images": pptx_extract_images,
                "excel_extract_images": excel_extract_images,
                "excel_max_rows": excel_max_rows,
                "use_image_resize": use_image_resize,
                "max_new_tokens": max_new_tokens,
                "image_batch_size": image_batch_size,
                "ocr_batch_size": ocr_batch_size
            }
            logger.info(f"Parser接收参数: {callin_args}")
            if not file:
                errorInfo = f"{anystatus.request_params_fail.msg}: 文件不存在."
                output = anyresponse.fail(
                    key = anystatus.request_params_fail.key,
                    code = anystatus.request_params_fail.code,
                    data = "",
                    msg = errorInfo                   
                )
                return output           
            
            ### 检查文件大小
            if (file_size / 1024 / 1024) >= float(maxsize):
                errorInfo = f"{anystatus.request_params_fail.msg}: 文件太大."
                output = anyresponse.fail(
                    key = anystatus.request_params_fail.key,
                    code = anystatus.request_params_fail.code,
                    data = "",
                    msg = errorInfo                   
                )
                return output           
            ### 检查文件类型
            allow_filetypes = set()
            for x,y in appviews.parser_model.filetypes_instance.model_dump().items():
                allow_filetypes.update(y)
            if file_suffix.lstrip('.').lower() not in allow_filetypes:
                errorInfo = f"{anystatus.request_params_fail.msg}: 文件类型不支持."
                output = anyresponse.fail(
                    key = anystatus.request_params_fail.key,
                    code = anystatus.request_params_fail.code,
                    data = "",
                    msg = errorInfo                   
                )
                return output  
            if file_suffix.lstrip('.').lower() not in appviews.parser_model.filetypes_instance['pdf']:
                stream = False
            
            new_uuid = uuid.uuid4().hex
            task_id = f"{appviews.seckey_prefix.task}-{new_uuid}"
            save_path = cache_dir.joinpath(f"files/{task_id}/upload")
            if not save_path.exists():
                save_path.mkdir(parents=True, exist_ok=True)
            save_file = save_path.joinpath(file_name)   
            save_infos_file = save_path.parent.joinpath("file_infos.json")             
            
            ### 开始下载
            try:
                with open(save_file,'wb') as f:
                    while chunk := await file.read(int(file_read_size)):
                        f.write(chunk)
                        
                file_infos = FileObject(
                    id = task_id,
                    object = "file",
                    bytes = file_size,
                    created_at = int(datetime.datetime.now().timestamp()),
                    filename = file_name,
                    purpose = "",
                    code = anystatus.success.code,
                    msg = anystatus.success.msg
                )
                with open(save_infos_file, 'w', encoding = 'utf-8') as f:
                    file_infos = file_infos.model_dump()
                    json.dump(file_infos, f, indent = 4, ensure_ascii = False)
                                    
            except:
                traceback.print_exc()
                shutil.rmtree(save_path.absolute().as_posix())
                errorInfo = f"{anystatus.request_params_fail.msg}: 文件下载失败."
                output = anyresponse.fail(
                    key = anystatus.request_params_fail.key,
                    code = anystatus.request_params_fail.code,
                    data = "",
                    msg = errorInfo                   
                )
                return output  
            parser_output = await appviews.parser_model.ainvoke(
                file = save_file,
                file_idx = task_id,
                autodetect_encoding = True,
                use_doc_cls = use_doc_cls,
                use_doc_rectifier = use_doc_rectifier,
                use_doc_layout = use_doc_layout,
                doc_layout_image_min_size = doc_layout_image_min_size,
                text_encoding = text_encoding,
                table_chunk_size = table_chunk_size,
                table_custom_separator = table_custom_separator,
                dpi = int(dpi),
                verbose = verbose,
                stream = stream,
                autocal_md5 = autocal_md5,
                docx_extract_headers_footers = docx_extract_headers_footers,
                docx_extract_images = docx_extract_images,
                pptx_extract_images = pptx_extract_images,
                excel_extract_images = excel_extract_images,
                excel_max_rows = excel_max_rows,
                use_image_resize = use_image_resize,
                max_new_tokens = max_new_tokens,
                image_batch_size = image_batch_size,
                ocr_batch_size = ocr_batch_size
            )
            parse_output_path = save_path.parent.joinpath("parser")
            if not parse_output_path.exists():
                parse_output_path.mkdir(parents=True, exist_ok=True)
            parse_output_file = parse_output_path.joinpath('parser_output.json')
            
            if not stream: 
                parser_output = parser_output.model_dump()
                with open(parse_output_file, 'w', encoding='utf-8') as f:
                    json.dump(parser_output,f,indent=4,ensure_ascii=False)

                output = anyresponse.success(
                    key = anystatus.success.key,
                    code = anystatus.success.code,
                    data = parser_output,
                    msg = anystatus.success.msg                   
                )   
            else:
                async def aevent_stream(parser_output):
                    async with aiofiles.open(parse_output_file, mode='a+', encoding='utf-8') as f:
                        async for item in parser_output:
                            item = item.model_dump_json()
                            await f.write(f"{item}\n")
                            yield item
                output = EventSourceResponse(aevent_stream(parser_output))
            return output     
            
        except Exception as e:
            errorInfo = traceback.format_exc()
            output = anyresponse.fail(
                key = anystatus.internal_fail.key,
                code = anystatus.internal_fail.code,
                data = "",
                msg = anystatus.internal_fail.msg              
            )
            return output
        
        finally:
            processTime = time.perf_counter() - rawTime
            appviews._call_api_log(
                callinfo = callinfo,
                argsTime = argsTime,
                processTime = processTime,
                errorInfo = errorInfo,
            )   
                

    return public_indexbp,protect_indexbp
