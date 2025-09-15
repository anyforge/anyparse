from abc import ABC


class parseCallback(ABC):

    def on_started(self, **kwargs):
        logtext = f"start file: {kwargs}."
        print(logtext)    
    

    def on_finished(self, **kwargs):
        logtext = f"finish file: {kwargs}."
        print(logtext)    
       
    def on_page_parsed(self, **kwargs):
        logtext = f"""finish page: {kwargs}."""
        print(logtext) 