from .base import AnyBaseError


class AnyFileNotFoundError(AnyBaseError):
    """
    文件不存在异常
    """
    def __init__(
        self,
        msg: str,
        code: int = 404,
    ):
        super().__init__(
            msg=msg,
            code=code
        )
        
        
class AnyFileTypeError(AnyBaseError):
    """
    文件类型异常
    """
    def __init__(
        self,
        msg: str,
        code: int = 415,
    ):
        super().__init__(
            msg=msg,
            code=code
        )
        

class AnyValueError(AnyBaseError):
    """
    值异常
    """
    def __init__(
        self,
        msg: str,
        code: int = 400,
    ):
        super().__init__(
            msg=msg,
            code=code
        )
        
        
class AnyRunTimeError(AnyBaseError):
    """
    运行时异常
    """
    def __init__(
        self,
        msg: str,
        code: int = 500,
    ):
        super().__init__(
            msg=msg,
            code=code
        )