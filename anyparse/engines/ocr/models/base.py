from abc import ABC,abstractmethod


class BaseOCRModel(ABC):
    @abstractmethod
    def invoke(self):
        pass
    
    @abstractmethod
    async def ainvoke(self):
        pass