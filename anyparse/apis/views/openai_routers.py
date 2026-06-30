import json
import uuid
import base64
import traceback
import time
import shutil
import aiofiles
import datetime
from pathlib import Path
from typing import Any, Union
from fastapi import Request,APIRouter, Form, File, UploadFile
from fastapi.responses import StreamingResponse, FileResponse
from .contexts import (
    mainviews, 
    get_appviews_config,
    get_cache_dir,
    default_maxsize,
    default_file_read_size,
)
from ..schemas.responses import anystatus
from ..schemas.openai import (
    ModelObject,
    ModelListResponse,
    
    # chat
    ChatMessageFileContentPart,
    ChatMessageContentPart,
    ChatMessage,
    ChatUsageInfo,
    ChatCompletionChoice,
    ChatCompletionResponse,
    ChatCompletionChunkDelta,
    ChatCompletionChunkChoice,
    ChatCompletionChunkResponse,
    ChatCompletionRequest,
    chat_format_openai_finish_chunk,
    chat_format_openai_stream_chunk,
    chat_format_openai_final_response,
    chat_format_openai_response_chunk,
    chat_format_openai_response_sse,
    
    # responses
    ResponsesMessageFileContentPart,
    ResponsesInputMessage,
    ResponsesAPIRequest,
    ResponsesOutputTextContent,
    ResponsesOutputMessage,
    ResponsesAPIResponse,
    ResponsesStreamEvent,
    ResponsesTextDeltaEvent,
    ResponsesTextDoneEvent,
    responses_format_openai_finish_chunk,
    responses_format_openai_stream_chunk,
    responses_format_openai_final_response,
    responses_format_openai_response_chunk,
    responses_format_openai_response_sse,
    
    # common
    FileObject,
    RuntimesArgs,
)
from ...loggers import logger


def create_openai_routers(public_indexbp: APIRouter, protect_indexbp: APIRouter):

    api_config = get_appviews_config()
    cache_dir = get_cache_dir()
    
    @protect_indexbp.get(
        path = api_config.get("modelapi",{}).get("openai_model_list", "/openai/v1/models"),
        summary = "OpenAI 兼容接口, 获取模型列表",
        response_model = ModelListResponse,
    )
    async def models_list_api(
        *,
        request: Request,
        appviews: Any = mainviews,
    ):
        argsTime = 0.0
        errorInfo = ""
        rawTime = time.perf_counter()
        try:
            callinfo = appviews._call_request_info(request)     
            allow_mimetypes = {}    
            model_mimetypes = appviews.parser_model.mimetypes_instance.model_dump()
            for x,y in model_mimetypes.items():
                allow_mimetypes[x] = y
            model_id = appviews.parser_model.model_id
            model_object = ModelObject(
                id = model_id,
                owned_by = model_id,
                allow_mimetypes = allow_mimetypes
            )
            models = [model_object]
            output = ModelListResponse(data=models)
            return output
            
        except Exception as e:
            errorInfo = traceback.format_exc()
            model_object = ModelObject(
                id = "Error",
                owned_by = "model",
                allow_mimetypes = {}
            )
            models = [model_object]
            output = ModelListResponse(data=models)
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
        path = api_config.get("modelapi",{}).get("openai_create_file", "/openai/v1/files"),
        summary = "OpenAI 兼容接口, 创建文件",
        response_model = FileObject,
    )
    async def create_file_api(
        file: UploadFile = File(...),
        purpose: str = Form(...),
        maxsize: float = Form(default_maxsize),
        file_read_size: int = Form(default_file_read_size),
        *,
        request: Request,
        appviews: Any = mainviews
    ):
        argsTime = 0.0
        errorInfo = ""
        rawTime = time.perf_counter()
        try:
            callinfo = appviews._call_request_info(request) 
            new_uuid = uuid.uuid4().hex
            file_id = f"{appviews.seckey_prefix.task}-{new_uuid}"
            request_purpose = f"{purpose}"
            file_name,file_size,file_suffix = file.filename,file.size,Path(file.filename).suffix
            if not file:
                errorInfo = f"{anystatus.request_params_fail.msg}: 文件不存在."
                output = FileObject(
                    id = file_id,
                    object = "file",
                    bytes = 0.0,
                    created_at = int(datetime.datetime.now().timestamp()),
                    filename = file_name,
                    purpose = request_purpose,
                    code = anystatus.request_params_fail.code,
                    msg = errorInfo
                )
                return output           
            
            ### 检查文件大小
            if (file_size / 1024 / 1024) >= float(maxsize):
                errorInfo = f"{anystatus.request_params_fail.msg}: 文件太大."
                output = FileObject(
                    id = file_id,
                    object = "file",
                    bytes = file_size,
                    created_at = int(datetime.datetime.now().timestamp()),
                    filename = file_name,
                    purpose = request_purpose,
                    code = anystatus.request_params_fail.code,
                    msg = errorInfo
                )
                return output           
            ### 检查文件类型
            allow_filetypes = set()
            for x,y in appviews.parser_model.filetypes_instance.model_dump().items():
                allow_filetypes.update(y)
            if file_suffix.lstrip('.').lower() not in allow_filetypes:
                errorInfo = f"{anystatus.request_params_fail.msg}: 文件类型不支持."
                output = FileObject(
                    id = file_id,
                    object = "file",
                    bytes = file_size,
                    created_at = int(datetime.datetime.now().timestamp()),
                    filename = file_name,
                    purpose = request_purpose,
                    code = anystatus.request_params_fail.code,
                    msg = errorInfo
                )
                return output              
            save_path = cache_dir.joinpath(f"files/{file_id}/upload")
            if not save_path.exists():
                save_path.mkdir(parents=True, exist_ok=True)
            save_file = save_path.joinpath(f"{file_name}")  
            save_infos_file = save_path.parent.joinpath("file_infos.json")            
            ### 开始下载
            try:
                with open(save_file,'wb') as f:
                    while chunk := await file.read(int(file_read_size)):
                        f.write(chunk)
                output = FileObject(
                    id = file_id,
                    object = "file",
                    bytes = file_size,
                    created_at = int(datetime.datetime.now().timestamp()),
                    filename = file_name,
                    purpose = request_purpose,
                    code = anystatus.success.code,
                    msg = anystatus.success.msg
                )
                with open(save_infos_file, 'w', encoding = 'utf-8') as f:
                    file_infos = output.model_dump()
                    json.dump(file_infos, f, indent = 4, ensure_ascii = False)
                    
                return output
            except:
                traceback.print_exc()
                shutil.rmtree(save_path.absolute().as_posix())
                errorInfo = f"{anystatus.request_params_fail.msg}: 文件下载失败."
                output = FileObject(
                    id = file_id,
                    object = "file",
                    bytes = file_size,
                    created_at = int(datetime.datetime.now().timestamp()),
                    filename = file_name,
                    purpose = request_purpose,
                    code = anystatus.request_params_fail.code,
                    msg = errorInfo
                )
                return output  
            
        except Exception as e:
            errorInfo = traceback.format_exc()
            output = FileObject(
                id = file_id,
                object = "file",
                bytes = 0.0,
                created_at = int(datetime.datetime.now().timestamp()),
                filename = file_name,
                purpose = "",
                code = anystatus.internal_fail.code,
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

    @protect_indexbp.get(
        path = api_config.get("modelapi",{}).get("openai_retrieve_file", "/openai/v1/files") + "/{file_id}",
        summary = "OpenAI 兼容接口, 获取文件元数据",
        response_model = FileObject,
    )
    async def retrieve_file_api(
        file_id: str,
        *,
        request: Request,
        appviews: Any = mainviews
    ):
        argsTime = 0.0
        errorInfo = ""
        rawTime = time.perf_counter()
        try:
            callinfo = appviews._call_request_info(request)                      
            save_path = cache_dir.joinpath(f"files/{file_id}/upload")
            save_infos_file = save_path.parent.joinpath("file_infos.json") 
            if not save_infos_file.exists():
                errorInfo = f"{anystatus.request_params_fail.msg}: 文件不存在."
                output = FileObject(
                    id = file_id,
                    object = "file",
                    bytes = 0.0,
                    created_at = int(datetime.datetime.now().timestamp()),
                    filename = "",
                    purpose = "",
                    code = anystatus.request_params_fail.code,
                    msg = errorInfo
                )            
            else:
                with open(save_infos_file, 'r', encoding = 'utf-8') as f:
                    file_infos = json.load(f)
                
                file_name = file_infos.get("filename", "")
                save_file = save_path.joinpath(file_name) 
                if not save_file.exists():
                    errorInfo = f"{anystatus.request_params_fail.msg}: 文件不存在."
                    output = FileObject(
                        id = file_id,
                        object = "file",
                        bytes = 0.0,
                        created_at = int(datetime.datetime.now().timestamp()),
                        filename = "",
                        purpose = "",
                        code = anystatus.request_params_fail.code,
                        msg = errorInfo
                    )                           
                else:
                    file_size = save_file.stat().st_size
                    output = FileObject(
                        id = file_id,
                        object = "file",
                        bytes = file_size,
                        created_at = int(datetime.datetime.now().timestamp()),
                        filename = file_name,
                        purpose = "",
                        code = anystatus.success.code,
                        msg = anystatus.success.msg
                    )
            return output
            
        except Exception as e:
            errorInfo = traceback.format_exc()
            output = FileObject(
                id = file_id,
                object = "file",
                bytes = 0.0,
                created_at = int(datetime.datetime.now().timestamp()),
                filename = file_name,
                purpose = "",
                code = anystatus.internal_fail.code,
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
                
    @protect_indexbp.get(
        path = api_config.get("modelapi",{}).get("openai_content_file", "/openai/v1/files") + "/{file_id}/content",
        summary = "OpenAI 兼容接口, 获取文件内容"
    )
    async def content_file_api(
        file_id: str,
        *,
        request: Request,
        appviews: Any = mainviews
    ):
        argsTime = 0.0
        errorInfo = ""
        rawTime = time.perf_counter()
        try:
            callinfo = appviews._call_request_info(request)                      
            save_path = cache_dir.joinpath(f"files/{file_id}/upload")
            save_infos_file = save_path.parent.joinpath("file_infos.json") 
            if not save_infos_file.exists():
                errorInfo = f"{anystatus.request_params_fail.msg}: 文件不存在."
                output = FileObject(
                    id = file_id,
                    object = "file",
                    bytes = 0.0,
                    created_at = int(datetime.datetime.now().timestamp()),
                    filename = "",
                    purpose = "",
                    code = anystatus.request_params_fail.code,
                    msg = errorInfo
                )            
            else:
                with open(save_infos_file, 'r', encoding = 'utf-8') as f:
                    file_infos = json.load(f)
                
                file_name = file_infos.get("filename", "")
                save_file = save_path.joinpath(file_name) 
                if not save_file.exists():
                    errorInfo = f"{anystatus.request_params_fail.msg}: 文件不存在."
                    output = FileObject(
                        id = file_id,
                        object = "file",
                        bytes = 0.0,
                        created_at = int(datetime.datetime.now().timestamp()),
                        filename = "",
                        purpose = "",
                        code = anystatus.request_params_fail.code,
                        msg = errorInfo
                    )                           
                else:
                    file_size = save_file.stat().st_size
                    file_infos = FileObject(
                        id = file_id,
                        object = "file",
                        bytes = file_size,
                        created_at = int(datetime.datetime.now().timestamp()),
                        filename = file_name,
                        purpose = "",
                        code = anystatus.success.code,
                        msg = anystatus.success.msg
                    )
                    return FileResponse(
                        path=save_file,
                        media_type="application/octet-stream",  # 强制作为二进制流下载
                        filename=save_file.name,               # 告诉客户端下载时的默认文件名
                        headers = {
                            "file_infos": file_infos.model_dump_json(),
                        }
                    )                   
                    
            return output
            
        except Exception as e:
            errorInfo = traceback.format_exc()
            output = FileObject(
                id = file_id,
                object = "file",
                bytes = 0.0,
                created_at = int(datetime.datetime.now().timestamp()),
                filename = file_name,
                purpose = "",
                code = anystatus.internal_fail.code,
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
                
    @protect_indexbp.delete(
        path = api_config.get("modelapi",{}).get("openai_delete_file", "/openai/v1/files") + "/{file_id}",
        summary = "OpenAI 兼容接口, 删除文件",
        response_model = FileObject,
    )
    async def delete_file_api(
        file_id: str,
        *,
        request: Request,
        appviews: Any = mainviews
    ):
        argsTime = 0.0
        errorInfo = ""
        rawTime = time.perf_counter()
        try:
            callinfo = appviews._call_request_info(request)                    
            save_path = cache_dir.joinpath(f"files/{file_id}/upload")
            save_infos_file = save_path.parent.joinpath("file_infos.json") 
            if not save_infos_file.exists():
                errorInfo = f"{anystatus.request_params_fail.msg}: 文件不存在."
                output = FileObject(
                    id = file_id,
                    object = "file",
                    bytes = 0.0,
                    created_at = int(datetime.datetime.now().timestamp()),
                    filename = "",
                    purpose = "",
                    code = anystatus.request_params_fail.code,
                    msg = errorInfo
                )            
            else:
                with open(save_infos_file, 'r', encoding = 'utf-8') as f:
                    file_infos = json.load(f)
                    file_name = file_infos.get("filename", "")
                    
                save_file = save_path.joinpath(file_name) 
                if not save_file.exists():
                    errorInfo = f"{anystatus.request_params_fail.msg}: 文件不存在."
                    output = FileObject(
                        id = file_id,
                        object = "file",
                        bytes = 0.0,
                        created_at = int(datetime.datetime.now().timestamp()),
                        filename = "",
                        purpose = "",
                        code = anystatus.request_params_fail.code,
                        msg = errorInfo
                    )                   
                else:
                    file_size = save_file.stat().st_size
                    shutil.rmtree(save_path.parent.absolute().as_posix())
                    output = FileObject(
                        id = file_id,
                        object = "file",
                        bytes = file_size,
                        created_at = int(datetime.datetime.now().timestamp()),
                        filename = file_name,
                        purpose = "",
                        deleted = True,
                        code = anystatus.success.code,
                        msg = anystatus.success.msg
                    )
            return output
            
        except Exception as e:
            errorInfo = traceback.format_exc()
            output = FileObject(
                id = file_id,
                object = "file",
                bytes = 0.0,
                created_at = int(datetime.datetime.now().timestamp()),
                filename = file_name,
                purpose = "",
                code = anystatus.internal_fail.code,
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
        path = api_config.get("modelapi",{}).get("openai_chat_completions", "/openai/v1/chat/completions"),
        summary = "OpenAI 兼容接口, 聊天接口",
        response_model = Union[ChatCompletionResponse, None],
    )
    async def chat_completions_api(
        chatrequest: ChatCompletionRequest,
        *,
        request: Request,
        appviews: Any = mainviews
    ):
        argsTime = 0.0
        errorInfo = ""
        rawTime = time.perf_counter()
        try:
            callinfo = appviews._call_request_info(request) 
            new_uuid = uuid.uuid4().hex
            request_id = f"chat-{new_uuid}"
            display_args = chatrequest.model_dump()
            logger.info(f"Parser接收参数: request_id: {request_id}, {display_args}") 
            
            model_id = appviews.parser_model.model_id
            model_id_list = [model_id.lower()]
            
            request_model = chatrequest.model
            request_stream = chatrequest.stream
            
            runtimes_args = chatrequest.runtimes_args or {}
            runtimes_args['stream'] = request_stream
            runtimes_args = RuntimesArgs(**runtimes_args)
            
            request_type = chatrequest.messages[0].content[0].type
            file_base64 = chatrequest.messages[0].content[0].file.file_data

            allow_filetypes = set()
            for x,y in appviews.parser_model.filetypes_instance.model_dump().items():
                allow_filetypes.update(y)

            if request_type not in ("file_id", "file", "text"):
                errorInfo = f"{anystatus.request_params_fail.msg}: messages_type: {request_type} 不允许."
                total_tokens = len(errorInfo)
                total_content = {
                    "code": anystatus.request_params_fail.code,
                    "msg": errorInfo,
                    "data": {}
                }
                if not runtimes_args.stream:
                    output = chat_format_openai_final_response(
                        total_content = total_content, 
                        request_id = request_id, 
                        model = model_id,
                        total_tokens = total_tokens
                    )
                    return output
                else:
                    output = chat_format_openai_response_sse(
                        total_content = total_content, 
                        request_id = request_id, 
                        model = model_id
                    )       
                    return output   

            # 模型校验
            if request_model.lower() not in model_id_list:
                errorInfo = f"{anystatus.request_params_fail.msg}: 模型 {chatrequest.model} 不存在."
                total_tokens = len(errorInfo)
                total_content = {
                    "code": anystatus.request_params_fail.code,
                    "msg": errorInfo,
                    "data": {}
                }
                if not runtimes_args.stream:
                    output = chat_format_openai_final_response(
                        total_content = total_content, 
                        request_id = request_id, 
                        model = model_id,
                        total_tokens = total_tokens
                    )
                    return output
                else:
                    output = chat_format_openai_response_sse(
                        total_content = total_content, 
                        request_id = request_id, 
                        model = model_id
                    )                
                    return output   
            # 文件类型校验
            if request_type != "file_id":
                allow_mimetypes = {}    
                model_mimetypes = appviews.parser_model.mimetypes_instance.model_dump()
                for x,y in model_mimetypes.items():
                    allow_mimetypes[x] = y
                
                file_suffix = ""
                file_mimetype = ""
                for k,v in allow_mimetypes.items():
                    if file_base64.startswith(v):
                        file_suffix = k
                        file_mimetype = v
                        break
                if not file_suffix:
                    errorInfo = f"{anystatus.request_params_fail.msg}: 文件mimetype不被支持."
                    total_tokens = len(errorInfo)
                    total_content = {
                        "code": anystatus.request_params_fail.code,
                        "msg": errorInfo,
                        "data": {}
                    }
                    if not runtimes_args.stream:
                        output = chat_format_openai_final_response(
                            total_content = total_content, 
                            request_id = request_id, 
                            model = model_id,
                            total_tokens = total_tokens
                        )
                        return output
                    else:
                        output = chat_format_openai_response_sse(
                            total_content = total_content, 
                            request_id = request_id, 
                            model = model_id
                        )                  
                        return output   

                task_id = f"{appviews.seckey_prefix.task}-{new_uuid}"
                file_bytes = base64.b64decode(file_base64.split(',')[1])
                file_size = len(file_bytes)
                file_name = f"{task_id}.{file_suffix}"
                ### 检查文件大小
                if (file_size / 1024 / 1024) >= float(runtimes_args.maxsize):
                    errorInfo = f"{anystatus.request_params_fail.msg}: 文件太大."
                    total_tokens = len(errorInfo)
                    total_content = {
                        "code": anystatus.request_params_fail.code,
                        "msg": errorInfo,
                        "data": {}
                    }
                    if not runtimes_args.stream:
                        output = chat_format_openai_final_response(
                            total_content = total_content, 
                            request_id = request_id, 
                            model = model_id,
                            total_tokens = total_tokens
                        )
                        return output     
                    else:
                        output = chat_format_openai_response_sse(
                            total_content = total_content, 
                            request_id = request_id, 
                            model = model_id
                        )                
                        return output   
                
                save_path = cache_dir.joinpath(f"files/{task_id}/upload")
                if not save_path.exists():
                    save_path.mkdir(parents=True, exist_ok=True)
                save_file = save_path.joinpath(file_name)   
                save_infos_file = save_path.parent.joinpath("file_infos.json")
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
                                        
                with open(save_file,'wb') as f:
                    f.write(file_bytes)     
            else:
                task_id = file_base64
                save_path = cache_dir.joinpath(f"files/{task_id}/upload")
            save_infos_file = save_path.parent.joinpath("file_infos.json") 
            if not save_infos_file.exists():
                errorInfo = f"{anystatus.request_params_fail.msg}: 文件不存在."
                total_tokens = len(errorInfo)
                total_content = {
                    "code": anystatus.request_params_fail.code,
                    "msg": errorInfo,
                    "data": {}
                }
                if not runtimes_args.stream:
                    output = chat_format_openai_final_response(
                        total_content = total_content, 
                        request_id = request_id, 
                        model = model_id,
                        total_tokens = total_tokens
                    )
                    return output     
                else:
                    output = chat_format_openai_response_sse(
                        total_content = total_content, 
                        request_id = request_id, 
                        model = model_id
                    )                
                    return output            
            else:
                with open(save_infos_file, 'r', encoding = 'utf-8') as f:
                    file_infos = json.load(f)
                
                file_name = file_infos.get("filename", "")
                save_file = save_path.joinpath(file_name)                
                file_suffix = save_file.suffix
                if not save_file.exists():
                    errorInfo = f"{anystatus.request_params_fail.msg}: 文件不存在."
                    total_tokens = len(errorInfo)
                    total_content = {
                        "code": anystatus.request_params_fail.code,
                        "msg": errorInfo,
                        "data": {}
                    }
                    if not runtimes_args.stream:
                        output = chat_format_openai_final_response(
                            total_content = total_content, 
                            request_id = request_id, 
                            model = model_id,
                            total_tokens = total_tokens
                        )
                        return output     
                    else:
                        output = chat_format_openai_response_sse(
                            total_content = total_content, 
                            request_id = request_id, 
                            model = model_id
                        )               
                        return output   

                if file_suffix.lstrip('.').lower() not in allow_filetypes:
                    errorInfo = f"{anystatus.request_params_fail.msg}: 文件类型不支持."
                    total_tokens = len(errorInfo)
                    total_content = {
                        "code": anystatus.request_params_fail.code,
                        "msg": errorInfo,
                        "data": {}
                    }
                    if not runtimes_args.stream:
                        output = chat_format_openai_final_response(
                            total_content = total_content, 
                            request_id = request_id, 
                            model = model_id,
                            total_tokens = total_tokens
                        )
                        return output     
                    else:
                        output = chat_format_openai_response_sse(
                            total_content = total_content, 
                            request_id = request_id, 
                            model = model_id
                        )                
                        return output                   
                    
                ### 检查文件大小
                file_size = save_file.stat().st_size
                if (file_size / 1024 / 1024) >= float(runtimes_args.maxsize):
                    errorInfo = f"{anystatus.request_params_fail.msg}: 文件太大."
                    total_tokens = len(errorInfo)
                    total_content = {
                        "code": anystatus.request_params_fail.code,
                        "msg": errorInfo,
                        "data": {}
                    }
                    if not runtimes_args.stream:
                        output = chat_format_openai_final_response(
                            total_content = total_content, 
                            request_id = request_id, 
                            model = model_id,
                            total_tokens = total_tokens
                        )
                        return output     
                    else:
                        output = chat_format_openai_response_sse(
                            total_content = total_content, 
                            request_id = request_id, 
                            model = model_id
                        )                  
                        return output   

            if not runtimes_args.file_idx:
                runtimes_args.file_idx = task_id
                
            parser_output = await appviews.parser_model.ainvoke(
                file = save_file,
                file_idx = runtimes_args.file_idx,
                autodetect_encoding = runtimes_args.autodetect_encoding,
                stream = runtimes_args.stream,
                use_doc_cls = runtimes_args.use_doc_cls,
                use_doc_rectifier = runtimes_args.use_doc_rectifier,
                use_doc_layout = runtimes_args.use_doc_layout,
                doc_layout_image_min_size = runtimes_args.doc_layout_image_min_size,
                text_encoding = runtimes_args.text_encoding,
                table_chunk_size = runtimes_args.table_chunk_size,
                table_custom_separator = runtimes_args.table_custom_separator,
                dpi = int(runtimes_args.dpi),
                verbose = runtimes_args.verbose,
                autocal_md5 = runtimes_args.autocal_md5,
                docx_extract_headers_footers = runtimes_args.docx_extract_headers_footers,
                docx_extract_images = runtimes_args.docx_extract_images,
                pptx_extract_images = runtimes_args.pptx_extract_images,
                excel_extract_images = runtimes_args.excel_extract_images,
                excel_max_rows = runtimes_args.excel_max_rows,
                use_image_resize = runtimes_args.use_image_resize,
                max_new_tokens = runtimes_args.max_new_tokens,
                image_batch_size = runtimes_args.image_batch_size,
                ocr_batch_size = runtimes_args.ocr_batch_size
            )
            parse_output_path = save_path.parent.joinpath("parser")
            if not parse_output_path.exists():
                parse_output_path.mkdir(parents=True, exist_ok=True)
            parse_output_file = parse_output_path.joinpath('parser_output.json')
            if not runtimes_args.stream:
                parser_output = parser_output.model_dump()
                with open(parse_output_file, 'w', encoding='utf-8') as f:
                    json.dump(parser_output,f,indent=4,ensure_ascii=False)

                total_content = {
                    "code": anystatus.success.code,
                    "msg": anystatus.success.msg,
                    "data": parser_output
                }
                total_tokens = len(parser_output.get("content", ""))
                output = chat_format_openai_final_response(
                    total_content = total_content, 
                    request_id = request_id, 
                    model = model_id,
                    total_tokens = total_tokens
                )
                
            else:
                ### 流式按照pdf和非pdf支持
                async def aevent_stream(parser_output):
                    try:
                        async with aiofiles.open(parse_output_file, mode='a+', encoding='utf-8') as f:
                            if file_suffix.lstrip('.').lower() not in appviews.parser_model.filetypes_instance['pdf']:
                                ### 非pdf
                                parser_output = parser_output.model_dump()
                                item = json.dumps(parser_output,ensure_ascii=False)
                                await f.write(f"{item}\n") 
                                total_content = {
                                    "code": anystatus.success.code,
                                    "msg": anystatus.success.msg,
                                    "data": parser_output
                                }              
                                chunk = ChatCompletionChunkDelta(
                                    role="assistant",
                                    content=total_content
                                )
                                content = chat_format_openai_stream_chunk(
                                    chunk = chunk, 
                                    request_id = request_id, 
                                    model = model_id
                                )
                                yield content                 
                            else:
                                async for item in parser_output:
                                    parser_item = item.model_dump()
                                    item = json.dumps(parser_item,ensure_ascii=False)
                                    await f.write(f"{item}\n")
                                    total_content = {
                                        "code": anystatus.success.code,
                                        "msg": anystatus.success.msg,
                                        "data": parser_item
                                    }              
                                    chunk = ChatCompletionChunkDelta(
                                        role="assistant",
                                        content=total_content
                                    )
                                    content = chat_format_openai_stream_chunk(
                                        chunk = chunk, 
                                        request_id = request_id, 
                                        model = model_id
                                    )
                                    yield content                            
                        # 发送结束标记
                        content = chat_format_openai_finish_chunk(
                                request_id = request_id, 
                                model = model_id
                        )
                        yield content
                        yield "data: [DONE]\n\n"

                    except Exception as e:
                        logger.error(f"Stream error: {traceback.format_exc()}")
                        yield "data: [DONE]\n\n"

                output = StreamingResponse(
                    aevent_stream(parser_output), 
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no"
                    }
                )                
                
            return output
            
        except Exception as e:
            errorInfo = traceback.format_exc()
            total_content = {
                "code": anystatus.internal_fail.code,
                "msg": anystatus.internal_fail.msg ,
                "data": {}
            }
            if not runtimes_args.stream:
                total_tokens = len(total_content['msg'])
                output = chat_format_openai_final_response(
                    total_content = total_content, 
                    request_id = request_id, 
                    model = model_id,
                    total_tokens = total_tokens
                )
                return output      
            else:
                output = chat_format_openai_response_sse(
                    total_content = total_content, 
                    request_id = request_id, 
                    model = model_id
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
        path = api_config.get("modelapi",{}).get("openai_responses", "/openai/v1/responses"),
        summary = "OpenAI 兼容接口, 聊天接口",
        response_model = Union[ResponsesAPIResponse, None],
    )
    async def reponses_completions_api(
        chatrequest: ResponsesAPIRequest,
        *,
        request: Request,
        appviews: Any = mainviews
    ):
        argsTime = 0.0
        errorInfo = ""
        rawTime = time.perf_counter()
        try:
            callinfo = appviews._call_request_info(request) 
            new_uuid = uuid.uuid4().hex
            request_id = f"resp-{new_uuid}"
            display_args = chatrequest.model_dump()
            logger.info(f"Parser接收参数: request_id: {request_id}, {display_args}") 
            
            model_id = appviews.parser_model.model_id
            model_id_list = [model_id.lower()]
            
            request_model = chatrequest.model
            request_stream = chatrequest.stream
            
            runtimes_args = chatrequest.runtimes_args or {}
            runtimes_args['stream'] = request_stream
            runtimes_args = RuntimesArgs(**runtimes_args)
            
            request_type = chatrequest.input[0].content[0].type
            file_base64 = chatrequest.input[0].content[0].file.file_data

            allow_filetypes = set()
            for x,y in appviews.parser_model.filetypes_instance.model_dump().items():
                allow_filetypes.update(y)

            if request_type not in ("file_id", "file", "text"):
                errorInfo = f"{anystatus.request_params_fail.msg}: messages_type: {request_type} 不允许."
                total_tokens = len(errorInfo)
                total_content = {
                    "code": anystatus.request_params_fail.code,
                    "msg": errorInfo,
                    "data": {}
                }
                if not runtimes_args.stream:
                    output = responses_format_openai_final_response(
                        total_content = total_content, 
                        request_id = request_id, 
                        model = model_id,
                        total_tokens = total_tokens,
                        status="failed"
                    )
                    return output
                else:
                    output = responses_format_openai_response_sse(
                        total_content = total_content, 
                        request_id = request_id, 
                        model = model_id
                    )       
                    return output   

            # 模型校验
            if request_model.lower() not in model_id_list:
                errorInfo = f"{anystatus.request_params_fail.msg}: 模型 {chatrequest.model} 不存在."
                total_tokens = len(errorInfo)
                total_content = {
                    "code": anystatus.request_params_fail.code,
                    "msg": errorInfo,
                    "data": {}
                }
                if not runtimes_args.stream:
                    output = responses_format_openai_final_response(
                        total_content = total_content, 
                        request_id = request_id, 
                        model = model_id,
                        total_tokens = total_tokens,
                        status="failed"
                    )
                    return output
                else:
                    output = responses_format_openai_response_sse(
                        total_content = total_content, 
                        request_id = request_id, 
                        model = model_id
                    )       
                    return output   
                 
            # 文件类型校验
            if request_type != "file_id":
                allow_mimetypes = {}    
                model_mimetypes = appviews.parser_model.mimetypes_instance.model_dump()
                for x,y in model_mimetypes.items():
                    allow_mimetypes[x] = y
                
                file_suffix = ""
                file_mimetype = ""
                for k,v in allow_mimetypes.items():
                    if file_base64.startswith(v):
                        file_suffix = k
                        file_mimetype = v
                        break
                if not file_suffix:
                    errorInfo = f"{anystatus.request_params_fail.msg}: 文件mimetype不被支持."
                    total_tokens = len(errorInfo)
                    total_content = {
                        "code": anystatus.request_params_fail.code,
                        "msg": errorInfo,
                        "data": {}
                    }
                    if not runtimes_args.stream:
                        output = responses_format_openai_final_response(
                            total_content = total_content, 
                            request_id = request_id, 
                            model = model_id,
                            total_tokens = total_tokens,
                            status="failed"
                        )
                        return output
                    else:
                        output = responses_format_openai_response_sse(
                            total_content = total_content, 
                            request_id = request_id, 
                            model = model_id
                        )       
                        return output   

                task_id = f"{appviews.seckey_prefix.task}-{new_uuid}"
                file_bytes = base64.b64decode(file_base64.split(',')[1])
                file_size = len(file_bytes)
                file_name = f"{task_id}.{file_suffix}"
                ### 检查文件大小
                if (file_size / 1024 / 1024) >= float(runtimes_args.maxsize):
                    errorInfo = f"{anystatus.request_params_fail.msg}: 文件太大."
                    total_tokens = len(errorInfo)
                    total_content = {
                        "code": anystatus.request_params_fail.code,
                        "msg": errorInfo,
                        "data": {}
                    }
                    if not runtimes_args.stream:
                        output = responses_format_openai_final_response(
                            total_content = total_content, 
                            request_id = request_id, 
                            model = model_id,
                            total_tokens = total_tokens,
                            status="failed"
                        )
                        return output
                    else:
                        output = responses_format_openai_response_sse(
                            total_content = total_content, 
                            request_id = request_id, 
                            model = model_id
                        )       
                        return output   
                
                save_path = cache_dir.joinpath(f"files/{task_id}/upload")
                if not save_path.exists():
                    save_path.mkdir(parents=True, exist_ok=True)
                save_file = save_path.joinpath(file_name)   
                save_infos_file = save_path.parent.joinpath("file_infos.json")
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
                                        
                with open(save_file,'wb') as f:
                    f.write(file_bytes)     
            else:
                task_id = file_base64
                save_path = cache_dir.joinpath(f"files/{task_id}/upload")
            save_infos_file = save_path.parent.joinpath("file_infos.json") 
            if not save_infos_file.exists():
                errorInfo = f"{anystatus.request_params_fail.msg}: 文件不存在."
                total_tokens = len(errorInfo)
                total_content = {
                    "code": anystatus.request_params_fail.code,
                    "msg": errorInfo,
                    "data": {}
                }
                if not runtimes_args.stream:
                    output = responses_format_openai_final_response(
                        total_content = total_content, 
                        request_id = request_id, 
                        model = model_id,
                        total_tokens = total_tokens,
                        status="failed"
                    )
                    return output
                else:
                    output = responses_format_openai_response_sse(
                        total_content = total_content, 
                        request_id = request_id, 
                        model = model_id
                    )       
                    return output            
            else:
                with open(save_infos_file, 'r', encoding = 'utf-8') as f:
                    file_infos = json.load(f)
                
                file_name = file_infos.get("filename", "")
                save_file = save_path.joinpath(file_name)                
                file_suffix = save_file.suffix
                if not save_file.exists():
                    errorInfo = f"{anystatus.request_params_fail.msg}: 文件不存在."
                    total_tokens = len(errorInfo)
                    total_content = {
                        "code": anystatus.request_params_fail.code,
                        "msg": errorInfo,
                        "data": {}
                    }
                    if not runtimes_args.stream:
                        output = responses_format_openai_final_response(
                            total_content = total_content, 
                            request_id = request_id, 
                            model = model_id,
                            total_tokens = total_tokens,
                            status="failed"
                        )
                        return output
                    else:
                        output = responses_format_openai_response_sse(
                            total_content = total_content, 
                            request_id = request_id, 
                            model = model_id
                        )       
                        return output   

                if file_suffix.lstrip('.').lower() not in allow_filetypes:
                    errorInfo = f"{anystatus.request_params_fail.msg}: 文件类型不支持."
                    total_tokens = len(errorInfo)
                    total_content = {
                        "code": anystatus.request_params_fail.code,
                        "msg": errorInfo,
                        "data": {}
                    }
                    if not runtimes_args.stream:
                        output = responses_format_openai_final_response(
                            total_content = total_content, 
                            request_id = request_id, 
                            model = model_id,
                            total_tokens = total_tokens,
                            status="failed"
                        )
                        return output
                    else:
                        output = responses_format_openai_response_sse(
                            total_content = total_content, 
                            request_id = request_id, 
                            model = model_id
                        )       
                        return output                
                    
                ### 检查文件大小
                file_size = save_file.stat().st_size
                if (file_size / 1024 / 1024) >= float(runtimes_args.maxsize):
                    errorInfo = f"{anystatus.request_params_fail.msg}: 文件太大."
                    total_tokens = len(errorInfo)
                    total_content = {
                        "code": anystatus.request_params_fail.code,
                        "msg": errorInfo,
                        "data": {}
                    }
                    if not runtimes_args.stream:
                        output = responses_format_openai_final_response(
                            total_content = total_content, 
                            request_id = request_id, 
                            model = model_id,
                            total_tokens = total_tokens,
                            status="failed"
                        )
                        return output
                    else:
                        output = responses_format_openai_response_sse(
                            total_content = total_content, 
                            request_id = request_id, 
                            model = model_id
                        )       
                        return output   

            if not runtimes_args.file_idx:
                runtimes_args.file_idx = task_id
                
            parser_output = await appviews.parser_model.ainvoke(
                file = save_file,
                file_idx = runtimes_args.file_idx,
                autodetect_encoding = runtimes_args.autodetect_encoding,
                stream = runtimes_args.stream,
                use_doc_cls = runtimes_args.use_doc_cls,
                use_doc_rectifier = runtimes_args.use_doc_rectifier,
                use_doc_layout = runtimes_args.use_doc_layout,
                doc_layout_image_min_size = runtimes_args.doc_layout_image_min_size,
                text_encoding = runtimes_args.text_encoding,
                table_chunk_size = runtimes_args.table_chunk_size,
                table_custom_separator = runtimes_args.table_custom_separator,
                dpi = int(runtimes_args.dpi),
                verbose = runtimes_args.verbose,
                autocal_md5 = runtimes_args.autocal_md5,
                docx_extract_headers_footers = runtimes_args.docx_extract_headers_footers,
                docx_extract_images = runtimes_args.docx_extract_images,
                pptx_extract_images = runtimes_args.pptx_extract_images,
                excel_extract_images = runtimes_args.excel_extract_images,
                excel_max_rows = runtimes_args.excel_max_rows,
                use_image_resize = runtimes_args.use_image_resize,
                max_new_tokens = runtimes_args.max_new_tokens,
                image_batch_size = runtimes_args.image_batch_size,
                ocr_batch_size = runtimes_args.ocr_batch_size
            )
            parse_output_path = save_path.parent.joinpath("parser")
            if not parse_output_path.exists():
                parse_output_path.mkdir(parents=True, exist_ok=True)
            parse_output_file = parse_output_path.joinpath('parser_output.json')
            if not runtimes_args.stream:
                parser_output = parser_output.model_dump()
                with open(parse_output_file, 'w', encoding='utf-8') as f:
                    json.dump(parser_output,f,indent=4,ensure_ascii=False)

                total_content = {
                    "code": anystatus.success.code,
                    "msg": anystatus.success.msg,
                    "data": parser_output
                }
                total_tokens = len(parser_output.get("content", ""))
                output = responses_format_openai_final_response(
                    total_content = total_content, 
                    request_id = request_id, 
                    model = model_id,
                    total_tokens = total_tokens,
                    status="completed"
                )
                
            else:
                ### 流式按照pdf和非pdf支持
                async def aevent_stream(parser_output):
                    try:
                        async with aiofiles.open(parse_output_file, mode='a+', encoding='utf-8') as f:
                            if file_suffix.lstrip('.').lower() not in appviews.parser_model.filetypes_instance['pdf']:
                                ### 非pdf
                                parser_output = parser_output.model_dump()
                                item = json.dumps(parser_output,ensure_ascii=False)
                                await f.write(f"{item}\n") 
                                total_content = {
                                    "code": anystatus.success.code,
                                    "msg": anystatus.success.msg,
                                    "data": parser_output
                                }              
                                content = responses_format_openai_stream_chunk(
                                    chunk_content=total_content, 
                                    request_id = request_id, 
                                    model = model_id
                                )
                                yield content                 
                            else:
                                async for item in parser_output:
                                    parser_item = item.model_dump()
                                    item = json.dumps(parser_item,ensure_ascii=False)
                                    await f.write(f"{item}\n")
                                    total_content = {
                                        "code": anystatus.success.code,
                                        "msg": anystatus.success.msg,
                                        "data": parser_item
                                    }              
                                    content = responses_format_openai_stream_chunk(
                                        chunk_content=total_content,
                                        request_id = request_id, 
                                        model = model_id
                                    )
                                    yield content                            
                        # 发送结束标记
                        content = responses_format_openai_finish_chunk(
                                request_id = request_id, 
                                model = model_id
                        )
                        yield content
                        yield "data: [DONE]\n\n"

                    except Exception as e:
                        logger.error(f"Stream error: {traceback.format_exc()}")
                        yield "data: [DONE]\n\n"

                output = StreamingResponse(
                    aevent_stream(parser_output), 
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no"
                    }
                )                
                
            return output
            
        except Exception as e:
            errorInfo = traceback.format_exc()
            total_content = {
                "code": anystatus.internal_fail.code,
                "msg": anystatus.internal_fail.msg ,
                "data": {}
            }
            total_tokens = len(total_content['msg'])
            if not runtimes_args.stream:
                output = responses_format_openai_final_response(
                    total_content = total_content, 
                    request_id = request_id, 
                    model = model_id,
                    total_tokens = total_tokens,
                    status="failed"
                )
                return output
            else:
                output = responses_format_openai_response_sse(
                    total_content = total_content, 
                    request_id = request_id, 
                    model = model_id
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