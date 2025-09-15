import re
import time
import traceback
from ..utils.utils import Readf


class textConverter(object):
    def __init__(self):
        pass
    
    def invoke_text(self,file,encoding='utf-8', **kwargs):
        try:
            tt = time.time()
            res = []
            content = ""
            content = [x for x in Readf(file,encoding=encoding,strip = False)]
            content = ''.join(content)
            res.append({
                "line_id": "1",
                "type": "text",
                "content": content,
                "time_elapse": time.time() - tt
            })
        except:
            traceback.print_exc()
            res = []
        finally:
            return res
            
            
class markdownConverter(object):
    def __init__(self):
        pass
    
    def invoke_markdown(self,file,encoding='utf-8', **kwargs):
        try:
            tt = time.time()
            res = []
            content = [x for x in Readf(file,encoding=encoding,strip = False)]
            content = ''.join(content)
            res.append({
                "line_id": "1",
                "type": "text",
                "content": content,
                "time_elapse": time.time() - tt
            })
        except:
            traceback.print_exc()
            res = []
        finally:
            return res

