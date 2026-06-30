import os
from abc import ABC, abstractmethod


class BaseConverter(ABC):
    @abstractmethod
    def invoke_item(self, **kwargs):
        return        
    
    @abstractmethod
    async def ainvoke_item(self, **kwargs):
        return        

    def invoke(self, file: str | os.PathLike, **kwargs):
        return self.invoke_item(file,**kwargs)
    
    async def ainvoke(self, file: str | os.PathLike, **kwargs):
        return await self.ainvoke_item(file,**kwargs)