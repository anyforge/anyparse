import gc
from abc import ABCMeta, abstractmethod
from fastapi import FastAPI,Request,APIRouter,Depends
from ...loggers import logger


class BaseViews(metaclass=ABCMeta):         

    def _call_request_info(self, request:Request):
        info = {
            "callPath": request.url.path,
            "callMethod": request.method,
            "callHeaders": request.headers,
            "callAddress": request.client.host
        } 
        return info
    
    def _call_api_log(
        self,
        callinfo: dict,
        argsTime: float,
        processTime: float,
        errorInfo: str = "",
        **kwargs
    ):
        log_format = (
            f"""{callinfo.get("callMethod")} - callPath: {callinfo.get("callPath")} - """
            f"""callAddress: {callinfo.get("callAddress")} - """
            f"""argsTime: {argsTime:.8f} - processTime: {processTime:.8f}"""
        )
        if kwargs:
            log_format += (
                f""" - extraInfo: {kwargs}"""
            )
        if not errorInfo:
            logger.info(log_format)
        else:
            errorInfo = f" - error: {errorInfo}"
            logger.error(log_format+errorInfo)
    
    def gc_collect(self, args):
        del args
        gc.collect()
    
    def get_rate_limit_key(self, request: Request):
        if not request.client or not request.client.host:
            return "127.0.0.1"

        return request.client.host        
        
    @abstractmethod
    def create_blueprint(self):
        pass