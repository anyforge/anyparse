from abc import ABC,abstractmethod
import datetime


class BaseCallback(ABC):
    @abstractmethod
    def on_started(self, **kwargs):
        nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logtext = f"[{nowtime}] start file: {kwargs}."
        print(logtext)    
    
    @abstractmethod
    def on_finished(self, **kwargs):
        nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logtext = f"[{nowtime}] finish file: {kwargs}."
        print(logtext)    
       
    @abstractmethod
    def on_page_started(self, **kwargs):
        nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logtext = f"[{nowtime}] start page: {kwargs}."
        print(logtext) 
       
    @abstractmethod
    def on_page_finished(self, **kwargs):
        nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logtext = f"""[{nowtime}] finish page: {kwargs}."""
        print(logtext)   