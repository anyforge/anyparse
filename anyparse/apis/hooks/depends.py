import traceback
from abc import ABCMeta, abstractmethod
from fastapi import Request
from ..schemas.responses import anystatus,anyresponse
from ...loggers import logger


class indexDenpends(metaclass=ABCMeta):                
    @abstractmethod
    async def __call__(self):
        pass


class UserAgentDenpends(indexDenpends):
    def __init__(self, **kwargs):
        self.root_api_key = kwargs.get("api_secret_key", "")
    
    async def __call__(self, request: Request):
        authorization = request.headers.get("Authorization", "")
        x_api_key = request.headers.get("x-api-key", "")
        logger.debug(f"request headers: {request.headers}")
        if not self.root_api_key:
            return True
        flag = False
        error_info = ""
        if authorization:
            try:
                scheme, token = authorization.split()
                if scheme.lower() != "bearer":
                    error_info = "Bearer is not found."
                    raise anyresponse.http_exception(
                        key = anystatus.request_auth_fail.key,
                        code = anystatus.request_auth_fail.code,
                        data = "",
                        msg = error_info
                    )
            except:
                traceback.print_exc()
                error_info = "Bearer format is not correct."
                raise anyresponse.http_exception(
                    key = anystatus.request_auth_fail.key,
                    code = anystatus.request_auth_fail.code,
                    data = "",
                    msg = error_info
                )
        else:
            # 情况2：如果没有 Authorization，则尝试从 X-API-Key 中获取（兼容模式）
            token = x_api_key
            if not token:
                # 两种鉴权头都没有，抛出异常
                error_info = "Authorization or X-API-Key is not found."
                raise anyresponse.http_exception(
                    key = anystatus.request_auth_fail.key,
                    code = anystatus.request_auth_fail.code,
                    data = "",
                    msg = error_info
                )

        if token.lower() != self.root_api_key.lower():
            error_info = "API-KEY is not valid."
            raise anyresponse.http_exception(
                key = anystatus.request_auth_fail.key,
                code = anystatus.request_auth_fail.code,
                data = "",
                msg = error_info
            )            
        else:
            flag = True
                
        return flag   