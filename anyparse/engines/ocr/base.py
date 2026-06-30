from abc import ABC,abstractmethod


class BaseOCRClient(ABC):
    @abstractmethod
    def invoke(self):
        pass
    
    @abstractmethod
    async def ainvoke(self):
        pass