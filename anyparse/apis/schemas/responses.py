from pathlib import Path
from typing import Any, Dict, Tuple, Optional, Union, AsyncGenerator
from pydantic import BaseModel
from sse_starlette import EventSourceResponse
from fastapi import HTTPException
from fastapi.responses import ORJSONResponse, FileResponse


ANY_HEADERS = {
    "content-type": "application/json",
    "server": "RestfulAPI",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "*",
    "X-Accel-Buffering": "no"
}


class ResponseStatus(BaseModel):
    key: str
    code: int
    msg: str


class AnyStatus:
    """
    合并后的响应状态管理器：
    - 可注册/更新状态（原 ResponseRegistry 功能）
    - 可通过属性访问状态（如 AnyStatus.success.code）
    """

    DEFAULT_STATUSES: Dict[str, Tuple[int, str]] = {
        "success": (2000, "success"),
        "failure": (2001, "failure"),
        "request_address_fail": (2002, "请求错误：一般为地址协议等"),
        "request_valid_fail": (2003, "请求内容错误：一般为参数等"),
        "request_params_fail": (2004, "请求参数错误"),
        "request_auth_fail": (2005, "请求认证错误"),
        "request_limiter_fail": (2006, "请求速率限制"),
        "internal_fail": (2007, "内部错误"),
    }

    def __init__(self, custom_statuses: Optional[Dict[str, Tuple[int, str]]] = None):
        self._statuses: Dict[str, ResponseStatus] = {}
        self._register_defaults()
        if custom_statuses:
            for key, (code, msg) in custom_statuses.items():
                self.register(key, code, msg)

    def _register_defaults(self):
        for key, (code, msg) in self.DEFAULT_STATUSES.items():
            self.register(key, code, msg)

    def register(self, key: str, code: int, msg: str):
        """注册一个状态"""
        self._statuses[key] = ResponseStatus(key = key, code = code, msg = msg)

    def get(self, key: str) -> Optional[ResponseStatus]:
        return self._statuses.get(key)

    def update(self, key: str, code: int = None, msg: str = None):
        if key not in self._statuses:
            raise KeyError(f"AnyStatus key '{key}' not registered.")
        status = self._statuses[key]
        if code is not None:
            status.code = code
        if msg is not None:
            status.msg = msg

    def dict(self) -> Dict[str, ResponseStatus]:
        return self._statuses.copy()

    # ========== 属性访问支持 ==========
    def __getattr__(self, name: str) -> ResponseStatus:
        status_obj = self._statuses.get(name)
        if status_obj is None:
            raise AttributeError(f"'AnyStatus' has no status '{name}'")
        return status_obj


class AnyResponse:
    def __init__(self, registry: AnyStatus):
        self.registry = registry

    def _build_content(
        self,
        key_or_code: Union[str, int],
        *,
        data: Any = "",
        extra_msg: str = "",
        code: Optional[int] = None
    ) -> Dict:
        if isinstance(key_or_code, str):
            status_obj = self.registry.get(key_or_code)
            if not status_obj:
                raise ValueError(f"Unknown response key: {key_or_code}")
            default_code = status_obj.code
            default_msg = status_obj.msg
        else:
            default_code = key_or_code
            default_msg = "Unknown error"

        final_code = code if code is not None else default_code
        final_msg = extra_msg or default_msg
        return {"code": final_code, "data": data, "msg": final_msg}

    def _make_json_response(self, content: Dict, status_code: int, headers: Optional[Dict]) -> ORJSONResponse:
        final_headers = ANY_HEADERS.copy()
        if headers:
            final_headers.update(headers)
        return ORJSONResponse(content=content, status_code=status_code, headers=final_headers)

    def dump(self, key: str, data: Any = "", msg: str = "") -> Dict:
        return self._build_content(key, data=data, extra_msg=msg)

    def success(
        self,
        *,
        key: str = "success",
        code: Optional[int] = None,
        data: Any = "", 
        msg: str = "", 
        status_code: int = 200,
        headers: Optional[Dict] = None
    ) -> ORJSONResponse:
        content = self._build_content(key, data=data, extra_msg=msg, code=code)
        return self._make_json_response(content, status_code, headers)

    def fail(
        self,
        *,
        key: str = "failure",
        code: Optional[int] = None,
        data: Any = "", 
        msg: str = "", 
        status_code: int = 200,
        headers: Optional[Dict] = None
    ) -> ORJSONResponse:
        content = self._build_content(key, data=data, extra_msg=msg, code=code)
        return self._make_json_response(content, status_code, headers)

    def http_exception(
        self, 
        *, 
        key: str, 
        code: Optional[int] = None,
        data: Any = "", 
        msg: str = "", 
        status_code: int = 200,
        headers: Optional[Dict] = None
    ) -> HTTPException:
        content = self._build_content(key, data=data, extra_msg=msg, code=code)
        final_headers = ANY_HEADERS.copy()
        if headers:
            final_headers.update(headers)
        return HTTPException(status_code=status_code, detail=content, headers=final_headers)

    def sse(
        self,
        contentstream: AsyncGenerator,
        media_type: str = "text/event-stream",
        headers: Optional[Dict] = None
    ) -> EventSourceResponse:
        final_headers = ANY_HEADERS.copy()
        final_headers.pop("content-type", None)
        if headers:
            final_headers.update(headers)
        return EventSourceResponse(contentstream, media_type=media_type, headers=final_headers)

    def file_stream(
        self,
        file_path: Union[str, Path],
        filename: Optional[str] = None,
        fileinfo: str = "",
        media_type: str = "application/octet-stream",
        headers: Optional[Dict] = None
    ) -> FileResponse:
        file_path = Path(file_path)
        if not filename:
            filename = file_path.name

        cors_headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        }
        if fileinfo:
            cors_headers["fileinfo"] = fileinfo
        if headers:
            cors_headers.update(headers)

        return FileResponse(
            path=file_path,
            filename=filename,
            media_type=media_type,
            headers=cors_headers
        )

# ==============================
# ✅ 核心：响应系统初始化类
# ==============================
class ResponseManager:
    def __init__(self, custom_statuses: Optional[Dict[str, tuple]] = None):
        self._status = AnyStatus(custom_statuses=custom_statuses)
        self._response = AnyResponse(self._status)  # AnyResponse 接收 AnyStatus 实例（它有 .get 方法）

    @property
    def response(self):
        return self._response

    @property
    def status(self):
        return self._status

# ==============================
# 🌍 全局单例
# ==============================
_response_manager = ResponseManager()
anyresponse = _response_manager.response
anystatus = _response_manager.status