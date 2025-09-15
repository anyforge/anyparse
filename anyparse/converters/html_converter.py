import re
import io
import time
import traceback
import html2text
import markdownify
from typing import Any, BinaryIO, Optional
from bs4 import BeautifulSoup
from urllib.parse import quote, unquote, urlparse, urlunparse
from ..utils.utils import Readf


class customMarkdownify(markdownify.MarkdownConverter):
    """
    A custom version of markdownify's MarkdownConverter. Changes include:

    - Altering the default heading style to use '#', '##', etc.
    - Removing javascript hyperlinks.
    - Truncating images with large data:uri sources.
    - Ensuring URIs are properly escaped, and do not conflict with Markdown syntax
    """

    def __init__(self, **options: Any):
        options["heading_style"] = options.get("heading_style", markdownify.ATX)
        options["keep_data_uris"] = options.get("keep_data_uris", True)
        # Explicitly cast options to the expected type if necessary
        super().__init__(**options)

    def convert_hn(
        self,
        n: int,
        el: Any,
        text: str,
        convert_as_inline: Optional[bool] = False,
        **kwargs,
    ) -> str:
        """Same as usual, but be sure to start with a new line"""
        if not convert_as_inline:
            if not re.search(r"^\n", text):
                return "\n" + super().convert_hn(n, el, text, convert_as_inline)  # type: ignore

        return super().convert_hn(n, el, text, convert_as_inline)  # type: ignore

    def convert_a(
        self,
        el: Any,
        text: str,
        convert_as_inline: Optional[bool] = False,
        **kwargs,
    ):
        """Same as usual converter, but removes Javascript links and escapes URIs."""
        prefix, suffix, text = markdownify.chomp(text)  # type: ignore
        if not text:
            return ""

        if el.find_parent("pre") is not None:
            return text

        href = el.get("href")
        title = el.get("title")

        # Escape URIs and skip non-http or file schemes
        if href:
            try:
                parsed_url = urlparse(href)  # type: ignore
                if parsed_url.scheme and parsed_url.scheme.lower() not in ["http", "https", "file"]:  # type: ignore
                    return "%s%s%s" % (prefix, text, suffix)
                href = urlunparse(parsed_url._replace(path=quote(unquote(parsed_url.path))))  # type: ignore
            except ValueError:  # It's not clear if this ever gets thrown
                return "%s%s%s" % (prefix, text, suffix)

        # For the replacement see #29: text nodes underscores are escaped
        if (
            self.options["autolinks"]
            and text.replace(r"\_", "_") == href
            and not title
            and not self.options["default_title"]
        ):
            # Shortcut syntax
            return "<%s>" % href
        if self.options["default_title"] and not title:
            title = href
        title_part = ' "%s"' % title.replace('"', r"\"") if title else ""
        return (
            "%s[%s](%s%s)%s" % (prefix, text, href, title_part, suffix)
            if href
            else text
        )

    def convert_img(
        self,
        el: Any,
        text: str,
        convert_as_inline: Optional[bool] = False,
        **kwargs,
    ) -> str:
        """Same as usual converter, but removes data URIs"""

        alt = el.attrs.get("alt", None) or ""
        src = el.attrs.get("src", None) or ""
        title = el.attrs.get("title", None) or ""
        title_part = ' "%s"' % title.replace('"', r"\"") if title else ""
        if (
            convert_as_inline
            and el.parent.name not in self.options["keep_inline_images_in"]
        ):
            return alt

        # Remove dataURIs
        if src.startswith("data:") and not self.options["keep_data_uris"]:
            src = src.split(",")[0] + "..."

        return "%s\n![%s](%s)" % (title_part, alt, src)

    def convert_soup(self, soup: Any) -> str:
        return super().convert_soup(soup)  # type: ignore
    
    
class htmlConverter(object):
    def __init__(self):
        pass
    
    def invoke_html(
            self,
            file,
            encoding='utf-8',
            **kwargs
        ):
        try:
            tt = time.time()
            res = []
            with open(file, "r", encoding=encoding) as file_stream:
                soup = BeautifulSoup(file_stream, "html.parser", from_encoding=encoding)
            # Remove javascript and style blocks
            for script in soup(["script", "style"]):
                script.extract()

            # Print only the main content
            body_elm = soup.find("body")
            webpage_text = ""
            if body_elm:
                webpage_text = customMarkdownify(**kwargs).convert_soup(body_elm)
            else:
                webpage_text = customMarkdownify(**kwargs).convert_soup(soup)

            assert isinstance(webpage_text, str)

            # remove leading and trailing \n
            markdown_text = webpage_text.strip()                
            res.append({
                "type": "html",
                "content": markdown_text,
                "time_elapse": time.time() - tt          
            })
        except:
            traceback.print_exc()
            res = []
        finally:
            return res   
        
    def invoke_string(self, html_string, **kwargs):
        try:
            tt = time.time()
            res = []
            soup = BeautifulSoup(html_string, "html.parser")
            # Remove javascript and style blocks
            for script in soup(["script", "style"]):
                script.extract()

            # Print only the main content
            body_elm = soup.find("body")
            webpage_text = ""
            if body_elm:
                webpage_text = customMarkdownify(**kwargs).convert_soup(body_elm)
            else:
                webpage_text = customMarkdownify(**kwargs).convert_soup(soup)

            assert isinstance(webpage_text, str)

            # remove leading and trailing \n
            markdown_text = webpage_text.strip()                
            res.append({
                "type": "html",
                "content": markdown_text,
                "time_elapse": time.time() - tt          
            })
        except:
            traceback.print_exc()
            res = []
        finally:
            return res   
        
    def invoke_string_v2(
            self,
            file,
            encoding='utf-8',
            strip = False,
            ignore_links = False,  # 忽略链接
            ignore_images = False, # 忽略图片
            ignore_emphasis = False, # 忽略强调（如加粗、斜体
            body_width = 78,  # 不限制输出宽度
            unicode_snob = False,  # 始终使用 Unicode 字符
            **kwargs
        ):
        try:
            tt = time.time()
            res = []
            content = [x for x in Readf(file,encoding=encoding,strip = strip)]
            content = ''.join(content)
            # 创建一个 Html2Text 对象
            h = html2text.HTML2Text()
            # 忽略转换过程中的样式
            h.ignore_links = ignore_links
            h.ignore_images = ignore_images
            h.ignore_emphasis = ignore_emphasis
            h.body_width = body_width  # 不限制输出宽度
            h.unicode_snob = unicode_snob  # 始终使用 Unicode 字符
            # 将 HTML 转换为 Markdown
            markdown_text = h.handle(content)
            res.append({
                "type": "html",
                "content": markdown_text,
                "time_elapse": time.time() - tt          
            })
        except:
            traceback.print_exc()
            res = []
        finally:
            return res   