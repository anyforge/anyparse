import os
import sys
import loguru
from pathlib import Path


class LoguruHandler(object):
    """
    desc:
        实例化日志回滚：loguru
        日志等级：DEBUG < INFO < WARNING < ERROR < CRITICAL
    args:
        filename: (type,str), 日志输出文件"./app.{time:YYYY-MM-DD}.log", 
        level = 'INFO', 日志等级
        encoding="utf-8",  文件编码
        rotation="00:00",  ### rotation='500 MB' # 在日志文件达到500MB，自动创建新的日志文件,rotation='00:00' # 每天00:00自动创建新的日志文件
        retention="30 days",mode='a+',  ### 日志保留天数
        colorize = True,    ### 颜色
        enqueue=True,   ### 多线程和多进程安全
        backtrace=True,diagnose=True,    ### 完整错误信息
        compression= 'zip', ###指定日志文件的压缩格式，compression= 'zip'
        format = 
    output:
        logger
    """
    def __init__(self, **kwargs):
        self.filename = kwargs.get('filename', None)
        self.level = kwargs.get('level', "DEBUG")
        self.encoding = kwargs.get('encoding', 'utf-8')
        self.mode = kwargs.get('mode', 'a+')
        self.rotation = kwargs.get('rotation', '00:00')
        self.retention = kwargs.get('retention', '30 days')
        self.colorize = kwargs.get('colorize', False)
        self.enqueue = kwargs.get('enqueue', True)
        self.backtrace = kwargs.get('backtrace', True)
        self.diagnose = kwargs.get('diagnose', True)
        self.compression = kwargs.get('compression', None)
        self.strformat = kwargs.get(
            'strformat', 
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{process}</cyan>:<cyan>{thread}</cyan> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>"
        )
        self.logger = loguru.logger
        self.logger.remove() # 移除默认
        self.console_handler_id = None
        self.file_handler_id = None
        self.__init_logger()
        
    def __init_logger(self):
        # 输出
        self.console_handler_id = self.logger.add(
            sys.stdout,
            level = self.level,
            colorize = True,
            enqueue = self.enqueue,
            backtrace = self.backtrace,
            # backtrace = True,
            diagnose = self.diagnose,
            format = self.strformat,
        )
        if self.filename:
            self.filename = Path(self.filename).expanduser().resolve()
            if not self.filename.parent.exists():
                self.filename.parent.mkdir(parents=True,exist_ok=True)
            self.file_handler_id = self.logger.add(
                self.filename,
                level = self.level,
                encoding = self.encoding,
                mode = self.mode,
                rotation = self.rotation,
                retention = self.retention,
                colorize = self.colorize,
                enqueue = self.enqueue,
                backtrace = self.backtrace,
                diagnose = self.diagnose,
                compression = self.compression,
                format = self.strformat
            )
        
    def set_level(
        self,
        level: str
    ):
        """
        动态修改日志等级。
        这会移除当前的处理器，并以新的等级重新添加。
        
        Args:
            level (str): 新的日志等级，如 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
        """
        if level.upper() not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            raise ValueError("Invalid log level, alolowed levels are: DEBUG, INFO, WARNING, ERROR, CRITICAL")
        
        self.level = level
        
        # 移除现有的处理器
        if self.console_handler_id is not None:
            self.logger.remove(self.console_handler_id)
        if self.file_handler_id is not None:
            self.logger.remove(self.file_handler_id)
        
        self.__init_logger()
        
    def add(
        self,
        sink,
        **kwargs
    ):
        """Add a handler sending log messages to a sink adequately configured.

            Parameters
            ----------
            sink : |file-like object|_, |str|, |Path|, |callable|_, |coroutine function|_ or |Handler|
                An object in charge of receiving formatted logging messages and propagating them to an
                appropriate endpoint.
        """
        if isinstance(sink, (str, os.PathLike)):
            sink = Path(sink).expanduser().resolve()
            if not sink.parent.exists():
                sink.parent.mkdir(parents=True,exist_ok=True)
        add_id = self.logger.add(sink, **kwargs)
        return add_id
    