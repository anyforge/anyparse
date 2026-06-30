import os
import time
import datetime
import asyncio
import traceback
import email
import email.policy
from email.message import EmailMessage
from email.utils import getaddresses, formataddr, parsedate_to_datetime
from typing import List, Optional
from bs4 import BeautifulSoup
from .base import BaseConverter
from .html import _CustomMarkdownify
from ..utils.utils import clean_text_linebreak


class EmailConverter(BaseConverter):
    """
    邮件解析器
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        
    def extract_addresses(self, msg: EmailMessage, header_name: str) -> Optional[List[str]]:
        header = msg.get(header_name)
        if not header: 
            return ""
        return [formataddr(addr) for addr in getaddresses([header])]

    def extract_metadata(self, msg: EmailMessage, **kwargs):
        metadata = {}
        metadata['subject'] = msg.get('Subject', '').strip()
        metadata['message_id'] = msg.get('Message-ID', '').strip().strip('<>')
        # 日期处理
        date_str = msg.get('Date')
        metadata['sent_date'] = ""
        if date_str:
            try:
                sent_dt = parsedate_to_datetime(date_str)
                metadata['sent_date'] = sent_dt.astimezone(
                    datetime.timezone.utc
                ).isoformat(timespec='seconds')
            except Exception:
                pass    
        metadata['from'] = self.extract_addresses(msg, 'From')
        metadata['to'] = self.extract_addresses(msg, 'To')
        metadata['cc'] = self.extract_addresses(msg, 'Cc')
        metadata['bcc'] = self.extract_addresses(msg, 'Bcc')
        return metadata
    
    def extract_body(
        self, 
        msg: EmailMessage,
        encoding='utf-8', 
        **kwargs
    ) -> dict:
        body_part = msg.get_body(preferencelist=('html', 'plain'))
        final_content = ""
        content_type = ""        
        if body_part is not None:
            content_type = body_part.get_content_type()
            # [unstructured 逻辑] 获取原始字节流并强制解码
            # get_payload(decode=True) 会处理 Base64/QP 编码，返回 bytes
            raw_payload = body_part.get_payload(decode=True)
            
            text_body = ""
            if isinstance(raw_payload, bytes):
                # 强制 UTF-8 解码，遇到错误替换为占位符
                text_body = raw_payload.decode('utf-8', errors='replace')
            elif isinstance(raw_payload, str):
                text_body = raw_payload
                
            # [分支处理]
            if content_type == "text/html":
                # 使用 markdownify 将 HTML 转为 Markdown
                soup = BeautifulSoup(text_body, "html.parser", from_encoding=encoding)
                for script in soup(["script", "style"]):
                    script.extract()

                # Print only the main content
                body_elm = soup.find("body")
                webpage_text = ""
                if body_elm:
                    webpage_text = _CustomMarkdownify(**kwargs).convert_soup(body_elm)
                else:
                    webpage_text = _CustomMarkdownify(**kwargs).convert_soup(soup)
                final_content = webpage_text.strip()             
            else:
                # text/plain 直接返回，无需转换
                final_content = text_body
            
        output = {
            "content": final_content,
            "content_type": content_type,
        }
        return output        
            
    def invoke_item(
        self,
        file: str | os.PathLike,
        ftype: str = "eml",
        encoding='utf-8', 
        **kwargs
    ) -> list:
        try:
            parse_callback = kwargs.get("parse_callback")
            parse_callback.on_started(**{
                "file": file
            })
            start_time = time.perf_counter()
            res = []
            
            with open(file, "rb") as f:
                msg: EmailMessage = email.message_from_binary_file(
                    f, 
                    policy=email.policy.default
                )
                        
            metadata = self.extract_metadata(msg, **kwargs)
            
            body_part = self.extract_body(msg, encoding=encoding, **kwargs)
            
            metadata.update({
                "content_type": body_part["content_type"],
            })
            
            content = body_part["content"]
            
            content = clean_text_linebreak(content)
            res.append({
                "line_id": "1",
                "type": f"{ftype}",
                "content": content,
                "title": metadata,
                "time_elapse": time.perf_counter() - start_time
            })            
            parse_callback.on_finished(**{
                "file": file
            })      
        except:
            traceback.print_exc()
            res = []
        finally:
            return res
        
    async def ainvoke_item(
        self,
        file: str | os.PathLike,
        ftype: str = "eml",
        encoding='utf-8', 
        **kwargs
    ) -> list:
        res = await asyncio.to_thread(
            self.invoke_item, 
            file, 
            ftype, 
            encoding, 
            **kwargs
        )
        return res