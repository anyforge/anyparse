import typer
from abc import ABC, abstractmethod
from rich.console import Console
from rich.panel import Panel
from rich.text import Text


class BaseCLI(ABC):
    """CLI 基础抽象类，封装通用逻辑"""

    def __init__(
        self,
        context_settings: dict = {"help_option_names": ["--help","-h"]},
        add_completion: bool = False,
        **kwargs
    ):
        self._app = typer.Typer(
            context_settings = context_settings,
            add_completion = add_completion,
            **kwargs
        )
        self.console = Console()
        # 注册子类定义的具体命令
        self.register_commands()
        self.app.command(
            name="version", 
            help="Show version information"
        )(self.show_version)
        # 注册基类自带的隐藏命令
        self.app.command(
            name="hidden", 
            hidden=True
        )(self._hidden_command)
        # 启动时展示 Logo
        self._show_logo()

    @property
    def app(self) -> typer.Typer:
        """获取 Typer 实例"""
        return self._app

    def _hidden_command(self):
        """这个命令是隐藏的，仅用于触发Typer显示命令列表，
        因为只有一个命令时，Typer会自动隐藏命令列表"""
        pass

    def _show_logo(self):
        """展示启动 Logo"""
        ascii_text = Text(text=self.logo, style="bold dark_cyan")
        self.console.print(ascii_text)
    
    def show_version(self):
        """显示版本信息"""
        version_info = (
            f"[bold green]Version:[/bold green] {self.version}\n"
        )

        self.console.print(
            Panel(
                version_info, 
                title = "[bold]System Info[/bold]", 
                subtitle = self.name,
                border_style = "cyan", 
                width=60
            )
        )

    @abstractmethod
    def register_commands(self):
        """
        子类必须实现此方法，用于注册具体的业务命令。
        """
        pass

    def run(self):
        """启动 CLI 的统一入口"""
        self._app()
