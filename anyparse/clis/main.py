import warnings
warnings.filterwarnings("ignore")
import json
import typer
from typing import Annotated
from rich import print
from pathlib import Path
from .base import BaseCLI
from .. import __version__
from ..loggers import anyparse_logger
from ..schemas import FileTypes, AnyConfig


def _set_log_level(log_level: str):
    """日志级别设置回调"""
    if log_level:
        anyparse_logger.set_level(log_level)
            

class AnyParseCLI(BaseCLI):
    def __init__(
        self, 
        context_settings: dict = {"help_option_names": ["--help","-h"]},
        add_completion: bool = False,
        **kwargs
    ):
        super().__init__(
            context_settings = context_settings,
            add_completion = add_completion,
            **kwargs
        )
        
    @property
    def name(self) -> str:
        return "AnyParse CLI"
        
    @property
    def logo(self) -> str:
        return __version__.logo
    
    @property
    def version(self) -> str:
        return __version__.version
    
    @staticmethod
    def _get_allow_filetypes(config: dict) -> set:
        """获取允许的文件类型集合"""
        total_filetypes = FileTypes(**config).model_dump()
        allow_filetypes = set()
        for x, y in total_filetypes.items():
            allow_filetypes.update(y)
        return total_filetypes,allow_filetypes

    def register_commands(self):
        """将所有命令方法注册到 Typer 实例中"""
        self.app.command(
            name="allow", 
            help="Show allow file suffix list"
        )(self.show_allowlist)
        self.app.command(
            name="parse", 
            help="Extract content from file"
        )(self.run_extract)
        self.app.command(
            name="api", 
            help="Parse file api"
        )(self.run_webapis)
        self.app.command(
            name="download", 
            help="Download models or files"
        )(self.run_downloads)
        
    def show_allowlist(
        self,
        config: Annotated[
            Path,
            typer.Option(
                "--config", "-c", exists=True, file_okay=True, dir_okay=False,
                readable=True, resolve_path=True, show_default=True,
                help="Path to local config yaml file",
            ),
        ],
    ):
        """显示允许的文件类型列表"""
        from rich.console import Console
        from rich.table import Column, Table
        from rich import box
        configfile = Path(config)
        if configfile.suffix.strip('.').lower() != 'yaml':
            print(
                f"[bold magenta]ERROR: 配置文件 {configfile.name} 不是 yaml 文件![/bold magenta]"
            )
            return

        ocr_config = AnyConfig.from_file(configfile).to_dict()
        total_filetypes,allow_filetypes = self._get_allow_filetypes(ocr_config['filetypes'])
        console = Console()
        # table = Table(show_header=True, header_style="bold magenta")
        table = Table(
            title="📂 允许的文件类型配置", 
            title_justify="left",
            show_header=True, 
            header_style="bold magenta",
            box=box.ROUNDED,      # 使用圆角边框，看起来更现代
            show_lines=True,      # 显示行之间的网格线，数据更清晰
            show_edge=True,
            title_style="bold cyan",
            padding=(0, 1)        # 单元格内的左右留白，避免文字紧贴边框
        )
        table.add_column("Name", style="bold yellow", min_width=10)
        table.add_column("Format", style="cyan", min_width=30)
        for k,v in total_filetypes.items():
            table.add_row(
                k, ', '.join(v)
            )

        console.print(table)
        return
    
    def run_downloads(
        self,
        config: Annotated[
            Path,
            typer.Option(
                "--config", "-c", exists=True, file_okay=True, dir_okay=False,
                readable=True, resolve_path=True, show_default=True,
                help="Path to local config yaml file",
            ),
        ],
        download_models: Annotated[
            bool,
            typer.Option(
                "--model", "-m",
                help = "Download models",
                show_default=True,
            )
        ] = False,
        
    ):
        configfile = Path(config)
        if configfile.suffix.strip('.').lower() != 'yaml':
            print(
                f"[bold magenta]ERROR: 配置文件 {configfile.name} 不是 yaml 文件![/bold magenta]"
            )
            return

        ocr_config = AnyConfig.from_file(configfile).to_dict()
        if not download_models:
            print(f"无需下载模型")
            pass
        else:
            from ..utils.downloads import download_anyparse_models
            config = ocr_config.get("anyparse", {})
            
            # doc cls
            model_path = config.get("doc_cls", {}).get("model_path", "")
            if model_path:
                model_path = Path(model_path).expanduser().resolve()
                model_path.mkdir(parents=True, exist_ok=True)
                download_anyparse_models(model_path)
                
            # doc rec
            model_path = config.get("doc_rectifier", {}).get("model_path", "")
            if model_path:
                model_path = Path(model_path).expanduser().resolve()
                model_path.mkdir(parents=True, exist_ok=True)
                download_anyparse_models(model_path)
                
            # doc layout
            model_path = config.get("layout", {}).get("model_path", "")
            if model_path:
                model_path = Path(model_path).expanduser().resolve()
                model_path.mkdir(parents=True, exist_ok=True)
                download_anyparse_models(model_path)
                
            # ocr model
            model_type = config.get("vlm", {}).get("model_type", "")
            model_path = config.get("vlm", {}).get(model_type, {}).get("model_path","")
            if model_path:
                model_path = Path(model_path).expanduser().resolve()
                model_path.mkdir(parents=True, exist_ok=True)
                download_anyparse_models(model_path)
        return

    def run_extract(
        self,
        config: Annotated[
            Path,
            typer.Option(
                "--config", "-c", exists=True, file_okay=True, dir_okay=False,
                readable=True, resolve_path=True, show_default=True,
                help="Path to local config yaml file",
            ),
        ],
        file: Annotated[
            Path,
            typer.Option(
                "--file", "-f", exists=True, file_okay=True, dir_okay=False,
                readable=True, resolve_path=True, show_default=True,
                help="Path to local file",
            ),
        ],
        log_level: Annotated[
            str,
            typer.Option(
                "--log-level", callback=_set_log_level,
                help="Set global log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
                show_default=True,
            ),
        ] = "DEBUG",
    ):
        """从文件中提取内容"""
        from ..pipelines import AnyParser
        
        configfile = Path(config)
        if configfile.suffix.strip('.').lower() != 'yaml':
            print(
                f"[bold magenta]ERROR: 配置文件 {configfile.name} 不是 yaml 文件![/bold magenta]"
            )
            return

        ocr_config = AnyConfig.from_file(configfile).to_dict()
        file = Path(file)
        file_suffix = file.suffix.strip('.').lower()
        total_filetypes,allow_filetypes = self._get_allow_filetypes(ocr_config['filetypes'])

        if file_suffix not in allow_filetypes:
            print(
                f"[bold magenta]ERROR: 文件类型 {file_suffix} 不被支持!, 仅支持: {ocr_config['filetypes']}.[/bold magenta]"
            )
            return

        parse_model = AnyParser(config=ocr_config)
        res = parse_model.invoke(file)
        print("[bold magenta]SUCCESS:[/bold magenta]", res.model_dump())

    def run_webapis(
        self,
        config: Annotated[
            Path,
            typer.Option(
                "--config", "-c", exists=True, file_okay=True, dir_okay=False,
                readable=True, resolve_path=True, show_default=True,
                help="Path to local config yaml file",
            ),
        ],
        host: Annotated[
            str, 
            typer.Option(
                "--host", "-h", 
                show_default=True, help="API host address"
            )
        ] = "0.0.0.0",
        port: Annotated[
            int, 
            typer.Option(
                "--port", "-p", 
                min=1000, max=65535, 
                show_default=True, help="API port number"
            )
        ] = 18007,
        workers: Annotated[
            int | None, 
            typer.Option(
                "--workers", "-w", 
                show_default=True, help="API workers number"
            )
        ] = None,
        prefix: Annotated[
            str, 
            typer.Option(
                "--prefix", "-x", 
                show_default=True, help="API prefix"
            )
        ] = "",
        seckey: Annotated[
            str, 
            typer.Option(
                "--seckey", "-s", 
                show_default=True, help="API secret key"
            )
        ] = "",
        api_extra_args: Annotated[
            str, 
            typer.Option(
                "--api-extra-args", 
                show_default=True, help="Extra args for API"
            )
        ] = "",
        log_level: Annotated[
            str,
            typer.Option(
                "--log-level", callback=_set_log_level,
                help="Set global log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
                show_default=True,
            ),
        ] = "DEBUG",
    ):
        """启动解析 API 服务"""
        from ..apis.app import AnyParserApi
        
        kwargs = {}
        try:
            if api_extra_args:
                kwargs = json.loads(api_extra_args)
        except Exception:
            pass

        parser_views = AnyParserApi(
            api_prefix=prefix,
            api_secret_key=seckey,
            config_file=config,
        )
        parser_views.run_app(
            host=host,
            port=port,
            workers=workers,
            server_header=False,
            **kwargs
        )
        

def anyparse_cli_main(**kwargs):
    cli = AnyParseCLI(**kwargs)
    cli.run()