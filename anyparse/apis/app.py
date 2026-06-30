import os
import copy
import datetime
import uvicorn
from pathlib import Path
from fastapi import FastAPI,Request,APIRouter,Depends
from .schemas.responses import anystatus,anyresponse
from .hooks.handlers import AnyHandler
from .hooks.depends import UserAgentDenpends
from .views.base import BaseViews
from .views.contexts import set_appviews_instance,KeyPrefix
from .views.parser_routers import create_parser_routers
from .views.openai_routers import create_openai_routers
from ..pipelines import AsyncAnyParser
from ..loggers import anyparse_logger,logger
from ..schemas import AnyConfig


class AnyParserApi(BaseViews):
    def __init__(
        self,
        api_secret_key: str = "",
        api_prefix: str = "",
        config_file: str | os.PathLike = "",
    ):
        self.api_config = {}
        if config_file:
            self.api_config = AnyConfig.from_file(config_file).to_dict()
        # 注册日志
        self.logger_id = self.init_logger(self.api_config.get('logger', {}))
        self.api_secret_key = api_secret_key or self.api_config['auths']['secret_key']
        self.api_prefix = api_prefix or self.api_config['modelapi']['prefix']
        self.anystatus = anystatus
        self.anyresponse = anyresponse
        self.seckey_prefix = KeyPrefix()
        self._app = self.create_app()
        
    @property
    def app(self):
        return self._app
    
    def init_logger(self, config: dict = {}):
        logger_id = None
        log_filename = config.get("filename", "")
        if log_filename:
            log_file = Path(log_filename).expanduser().resolve()
            # 创建日志目录
            if not log_file.parent.exists():
                log_file.parent.mkdir(parents=True, exist_ok=True)
            # 生成带时间戳的日志文件名
            log_time = datetime.datetime.now().strftime("%Y-%m-%d")
            log_filename = f"{log_file.stem}.{log_time}.{log_file.suffix.replace('.', '')}"
            log_filename = log_file.parent.joinpath(log_filename)
            loguru_config = config.get("loguru", {})
            add_kwargs = {
                "level": config.get("level", "INFO"),
                "encoding": loguru_config.get("encoding", "utf-8"),
                "mode": loguru_config.get("mode", "a+"),
                "rotation": loguru_config.get("rotation", "00:00"),
                "retention": loguru_config.get("retention", "30 days"),
                "colorize": loguru_config.get("colorize", False),
                "enqueue": loguru_config.get("enqueue", True),
                "backtrace": loguru_config.get("backtrace", True),
                "diagnose": loguru_config.get("diagnose", True),
                "compression": loguru_config.get("compression", None),
                "format": loguru_config.get("strformat", None)
            }
            add_kwargs = {k: v for k, v in add_kwargs.items() if v is not None}
            logger_id = anyparse_logger.add(
                log_filename,
                **add_kwargs
            )  
            logger.level("INFO", color="<green><bold>")
        return logger_id
          
    def init_model(self, config: dict = {}):
        self.parser_model = AsyncAnyParser(
            model_id = config.get("model_id", ""),
            config = config
        )

    def create_blueprint(self):
        dependencies = [
            Depends(
                dependency = UserAgentDenpends(
                    api_secret_key = self.api_secret_key
                )
            )
        ]
        public_indexbp = APIRouter(
            prefix='', 
            tags = ['public']
        )
        protect_indexbp  = APIRouter(
            prefix = '', 
            tags = ['protect'], 
            dependencies = dependencies
        )
        
        create_parser_routers(
            public_indexbp = public_indexbp, 
            protect_indexbp = protect_indexbp
        )
        create_openai_routers(
            public_indexbp = public_indexbp, 
            protect_indexbp = protect_indexbp
        )
        
        return public_indexbp,protect_indexbp
    
    def create_app(self):
        parser_args = {
            "config": copy.deepcopy(self.api_config)
        }
        anyhandler = AnyHandler()
        anyhandler.add_startup_hook(
            func = self.init_model,
            args = parser_args
        )
        app = FastAPI(
            lifespan = anyhandler.lifespans
        )
        anyhandler.invoke(app)
        # 👇 注册当前实例（关键一行）
        set_appviews_instance(self)
        # 注册路由
        public_indexbp,protect_indexbp = self.create_blueprint()
        app.include_router(
            public_indexbp,
            prefix = self.api_prefix
        )
        app.include_router(
            protect_indexbp, 
            prefix = self.api_prefix
        )
        return app     
    
    def run_app(
        self,
        host: str = "0.0.0.0",
        port: int = 18007,
        workers: int | None = None,
        server_header: bool = False,
        **kwargs
    ):
        host = host or self.api_config['modelapi']['host']
        port = port or int(self.api_config['modelapi']['port'])
        uvicorn.run(
            app = self.app,
            host = f"{host}",
            port = int(port),
            workers = workers,
            server_header = server_header,
            **kwargs
        )         