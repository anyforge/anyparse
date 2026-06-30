class AnyBaseError(Exception):
    """
    基类异常
    """
    def __init__(
        self,
        msg: str = "Unknown Error",
        code: int = 500,
    ):
        self._code = code
        self._msg = msg
        super().__init__(self.__str__())

    @property
    def name(self):
        return self.__class__.__name__
        
    @property
    def code(self):
        return self._code
    
    @property
    def msg(self):
        return self._msg
    
    def __str__(self):
        content = f"{self.name}(code={self.code}, msg={self.msg})"
        return content