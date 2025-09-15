import aiohttp
import requests as rq
from pathlib import Path


class anyFiled(object):
    def __init__(self):
        pass
    
    def detect_filename_from_url(self,url):
        try:
            urlpath = Path(url)
            name,stem,suffix = urlpath.name,urlpath.stem,urlpath.suffix
        
        except:
            name,stem,suffix = None,None,None
        
        finally:
            return (name,stem,suffix)
        
    def download_file(self, url, save_path):
        try:
            flag = True
            response = rq.get(url)
            with open(save_path, 'wb') as file:
                file.write(response.content)
        except:
            flag = False
            
        finally:
            return flag
    
    def download_large_file(self, url, save_path, chunk_size=4096):
        try:
            flag = True
            response = rq.get(url, stream=True)
            with open(save_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    file.write(chunk) 
        except:
            flag = False
        
        finally:
            return flag
        
    def download_file_with_token(self):
        try:
            flag = True
        except:
            flag = False
        
        finally:
            return flag      
        
    async def async_download_large_file(self, url, save_path, chunk_size=4096):
        try:
            flag = True
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    with open(save_path, 'wb') as fd:
                        # iter_chunked() 设置每次保存文件内容大小，单位bytes
                        async for chunk in resp.content.iter_chunked(chunk_size):
                            fd.write(chunk)
        
        except:
            flag = False
            
        finally:
            return flag