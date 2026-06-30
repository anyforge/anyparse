import datetime
from .base import BaseCallback


class ParseCallback(BaseCallback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def on_started(self, **kwargs):
        nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logtext = f"[{nowtime}] start file: {kwargs}."
        print(logtext)    
    
    def on_finished(self, **kwargs):
        nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logtext = f"[{nowtime}] finish file: {kwargs}."
        print(logtext)    
       
    def on_page_started(self, **kwargs):
        nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logtext = f"[{nowtime}] start page: {kwargs}."
        print(logtext) 
       
    def on_page_finished(self, **kwargs):
        nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logtext = f"""[{nowtime}] finish page: {kwargs}."""
        print(logtext)     