from abc import ABC,abstractmethod


class BaseASRClient(ABC):
    @abstractmethod
    def invoke(self):
        pass
    
    @abstractmethod
    async def ainvoke(self):
        pass