import json
import inspect
import traceback
from abc import ABCMeta
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from slowapi.errors import RateLimitExceeded
from typing import Callable, List, Any
from ..schemas.responses import anystatus,anyresponse
from ..schemas.openai import (
    chat_format_openai_final_response,
    chat_format_openai_response_sse,
    responses_format_openai_final_response,
    responses_format_openai_response_sse
)
from ...loggers import logger


class AnyHandler(metaclass=ABCMeta):
    def __init__(self):
        # 注册表：存储可扩展的回调函数
        self.startup_hooks: List[Callable] = []
        self.shutdown_hooks: List[Callable] = []
        self.exception_handlers: List[tuple] = []  # (exception_class, handler_func)
        self.middlewares: List[tuple] = []         # (middleware_class, **options)
        # 👇 默认注册 CORS（开发友好）
        self.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
    def _valid_is_openai_request(self, request: Request) -> bool:
        url_path = request.url.path.lower()
        if "openai".lower() not in url_path:
            return False
        else:
            return True
        
    async def _build_openai_handler(
        self, 
        request: Request, 
        status_code: int, 
        status_detail: dict = {}
    ):
        url_path = request.url.path.lower()
        content_type = request.headers.get('content-type') or ''
        is_stream = False
        try:
            if 'multipart/form-data' in content_type:
                form = await request.form()
                is_stream = form.get("stream", False)
            else:
                body = await request.body()
                if body:
                    body_json = json.loads(body)
                    is_stream = body_json.get("stream", False)
        except Exception:
            traceback.print_exc()
        if not status_detail:
            if status_code == 404:
                errorInfo = f"{anystatus.request_address_fail.msg}"
                total_content = {
                    "code": anystatus.request_address_fail.code,
                    "msg": errorInfo,
                    "data": {}
                }            
            elif status_code == 422:
                errorInfo = f"{anystatus.request_valid_fail.msg}"
                total_content = {
                    "code": anystatus.request_valid_fail.code,
                    "msg": errorInfo,
                    "data": {}
                }
            elif status_code == 429:
                errorInfo = f"{anystatus.request_limiter_fail.msg}"
                total_content = {
                    "code": anystatus.request_limiter_fail.code,
                    "msg": errorInfo,
                    "data": {}
                }    
            else:
                errorInfo = f"{anystatus.internal_fail.msg}"
                total_content = {
                    "code": anystatus.internal_fail.code,
                    "msg": errorInfo,
                    "data": {}
                }    
        else:    
            if status_detail and isinstance(status_detail, dict):
                errorInfo = f"{status_detail.get('msg', anystatus.internal_fail.msg)}"
                total_content = {
                    "code": status_detail.get('code', anystatus.internal_fail.code),
                    "msg": errorInfo,
                    "data": status_detail.get('data', "")
                }                   
            else:
                errorInfo = f"{anystatus.internal_fail.msg}"
                total_content = {
                    "code": anystatus.internal_fail.code,
                    "msg": errorInfo,
                    "data": {}
                }                
        
        if not is_stream:
            if "v1/responses".lower() not in url_path:
                output = responses_format_openai_final_response(
                    total_content = total_content, 
                    request_id = "", 
                    model = "",
                    total_tokens = 0,
                    status="failed"
                )                
            else:
                output = chat_format_openai_final_response(
                    total_content = total_content, 
                    request_id = "", 
                    model = "",
                    total_tokens = 0
                )
            return ORJSONResponse(
                status_code=200,  # OpenAI 参数错误通常返回 400
                content=output.model_dump(),
            )
        else:
            if "v1/responses".lower() not in url_path:
                output = responses_format_openai_response_sse(
                    total_content = total_content, 
                    request_id = "", 
                    model = ""
                )                  
            else:
                output = chat_format_openai_response_sse(
                    total_content = total_content, 
                    request_id = "", 
                    model = ""
                )                
            return output           
        
    async def _rate_limit_handler(self, request: Request, exc: RateLimitExceeded):
        content_type = request.headers.get('content-type') or ''
        logger.error(f"status_code: {exc.status_code}, http exception: {exc}, headers: {request.headers}, callpath: {request.url.path}, content_type: {content_type}")
        if not self._valid_is_openai_request(request):
            output = ORJSONResponse(
                status_code=429,
                content={
                    "code": self.anyresp.request_limiter_fail.code,
                    "data": "",
                    "msg": self.anyresp.request_limiter_fail.msg
                },
                headers = exc.headers
            )
        else:
            output = await self._build_openai_handler(
                request = request,
                status_code = 429
            )
        return output

    async def _http_exception_handler(self, request: Request, exc: StarletteHTTPException):
        content_type = request.headers.get('content-type') or ''
        status_detail = exc.detail
        logger.error(f"status_code: {exc.status_code}, http exception: {exc}, headers: {request.headers}, callpath: {request.url.path}, content_type: {content_type}")
        
        if not self._valid_is_openai_request(request):
            if isinstance(status_detail, dict):
                code = status_detail.get('code', 201)
                if code == anystatus.request_auth_fail.code:
                    return anyresponse.fail(
                        code = status_detail.get('code', anystatus.request_auth_fail.code),
                        data = status_detail.get('data', ""),
                        msg = status_detail.get('msg', anystatus.request_auth_fail.msg),
                        headers=exc.headers
                    )
                elif code == anystatus.request_limiter_fail.code:
                    return anyresponse.fail(
                        code = status_detail.get('code', anystatus.request_limiter_fail.code),
                        data = status_detail.get('data', ""),
                        msg = status_detail.get('msg', anystatus.request_limiter_fail.msg),
                        headers=exc.headers
                    )  
                else:
                    return anyresponse.fail(
                        code = status_detail.get('code', anystatus.internal_fail.code),
                        data = status_detail.get('data', ""),
                        msg = status_detail.get('msg', anystatus.internal_fail.msg),
                        headers=exc.headers
                    )                      
            if exc.status_code == 404:
                return anyresponse.fail(
                    code = anystatus.request_address_fail.code,
                    data = "",
                    msg = anystatus.request_address_fail.msg,
                    headers=exc.headers
                )                  
            else:
                safe_msg = status_detail if isinstance(status_detail, str) else str(status_detail)
                return anyresponse.fail(
                    msg = safe_msg,
                    headers=exc.headers
                )     
            
        else:
            exc_detail = {}
            if isinstance(status_detail, dict):
                exc_detail = status_detail
            output = await self._build_openai_handler(
                request = request,
                status_code = 404,
                status_detail = exc_detail
            )
            return output                    

    async def _validation_exception_handler(self, request: Request, exc: RequestValidationError):
        content_type = request.headers.get('content-type') or ''
        logger.error(f"status_code: 422, http exception: {exc}, headers: {request.headers}, callpath: {request.url.path}, content_type: {content_type}")
        if not self._valid_is_openai_request(request):
            output = anyresponse.fail(
                code = anystatus.request_valid_fail.code,
                data = "",
                msg = anystatus.request_valid_fail.msg,
                headers=exc.headers
            )       
        else:    
            output = await self._build_openai_handler(
                request = request,
                status_code = 422
            )
        return output
                
    # ------------------ Lifespan 扩展 ------------------
    def add_startup_hook(self, func: Callable, args: Any):
        """注册启动时钩子"""
        self.startup_hooks.append([func, args])

    def add_shutdown_hook(self, func: Callable, args: Any):
        """注册关闭时钩子"""
        self.shutdown_hooks.append([func, args])

    @asynccontextmanager
    async def lifespans(self, app: FastAPI):
        """统一的 lifespan 上下文管理器，执行所有注册的钩子"""
        # 执行 startup hooks
        success_startup_hooks = []
        failed_startup_hooks = []
        try:
            for hook,args in self.startup_hooks:
                try:
                    if inspect.iscoroutinefunction(hook):
                        await hook(**args)
                    else:
                        hook(**args)
                    success_startup_hooks.append(hook.__name__)
                except Exception as e:
                    traceback.print_exc()
                    failed_startup_hooks.append(hook.__name__)
        except:
            logger.error(traceback.format_exc())
        logger.info(f"start hooks, success: {success_startup_hooks}, failed: {failed_startup_hooks}")
        yield
        # 执行 shutdown hooks（逆序）
        success_shutdown_hooks = []
        failed_shutdown_hooks = []
        try:
            for hook,args in reversed(self.shutdown_hooks):
                try:
                    if inspect.iscoroutinefunction(hook):
                        await hook(**args)
                    else:
                        hook(**args)
                    success_shutdown_hooks.append(hook.__name__)
                except Exception as e:
                    traceback.print_exc()
                    failed_shutdown_hooks.append(hook.__name__)
        except:
            logger.error(traceback.format_exc())
        logger.info(f"shutdown hooks, success: {success_shutdown_hooks}, failed: {failed_shutdown_hooks}")

    # ------------------ 异常处理器扩展 ------------------
    def add_exception_handler(self, exc_class, handler):
        """注册自定义异常处理器"""
        self.exception_handlers.append((exc_class, handler))

    def register_exception_handlers(self, app: FastAPI):
        """注册默认异常处理器（可被子类覆盖或补充）"""
        # 默认处理器也可以通过 add_exception_handler 注册
        self.add_exception_handler(RateLimitExceeded, self._rate_limit_handler)
        self.add_exception_handler(StarletteHTTPException, self._http_exception_handler)
        self.add_exception_handler(RequestValidationError, self._validation_exception_handler)

        # 绑定到 app
        success_add_exceptions = []
        for exc_class, handler in self.exception_handlers:
            app.add_exception_handler(exc_class, handler)
            success_add_exceptions.append(exc_class.__name__)
        logger.info(f"add exception handlers: {success_add_exceptions}")

    # ------------------ 中间件扩展 ------------------
    def add_middleware(self, middleware_class, **options):
        """注册中间件"""
        self.middlewares.append((middleware_class, options))

    def register_middlewares(self, app: FastAPI):
        """将所有中间件添加到 app"""
        success_add_middlewares = []
        for middleware_class, options in self.middlewares:
            app.add_middleware(middleware_class, **options)
            success_add_middlewares.append(middleware_class.__name__)
        logger.info(f"add middlewares: {success_add_middlewares}")

    # ------------------ 统一入口 ------------------
    def invoke(self, app: FastAPI):
        """将所有注册项应用到 FastAPI 实例"""
        self.register_exception_handlers(app)
        self.register_middlewares(app)
        logger.info(f"Invoke hookers success")
        
        
################# Examples ##################

# class MyCustomRegister(anyRegister):
#     def __init__(self):
#         super().__init__()
#         # 添加自定义 startup 钩子
#         self.add_startup_hook(self.init_redis)
#         self.add_shutdown_hook(self.close_redis)

#         # 添加自定义中间件
#         self.add_middleware(SomeCustomMiddleware, option1="value")

#         # 添加自定义异常处理器
#         self.add_exception_handler(CustomError, self.handle_custom_error)

#     async def init_redis(self, app: FastAPI):
#         print("Connecting to Redis...")
#         # 初始化逻辑

#     async def close_redis(self, app: FastAPI):
#         print("Closing Redis...")

#     async def handle_custom_error(self, request: Request, exc: CustomError):
#         return ORJSONResponse(status_code=500, content={"error": "Custom error!"})