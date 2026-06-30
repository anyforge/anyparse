import os
import re
import time
import traceback
import asyncio
import zipfile
from defusedxml import minidom
from xml.dom.minidom import Document
from bs4 import BeautifulSoup
import markdownify
from urllib.parse import quote, unquote, urlparse, urlunparse
from typing import Any, Dict, List, BinaryIO, Optional
from .base import BaseConverter
from ..utils.utils import clean_text_linebreak

    
class _CustomMarkdownify(markdownify.MarkdownConverter):

    def __init__(self, **options: Any):
        options["heading_style"] = options.get("heading_style", markdownify.ATX)
        options["keep_data_uris"] = options.get("keep_data_uris", False)
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
        src = el.attrs.get("src", None) or el.attrs.get("data-src", None) or ""
        title = el.attrs.get("title", None) or ""
        title_part = ' "%s"' % title.replace('"', r"\"") if title else ""
        # Remove all line breaks from alt
        alt = alt.replace("\n", " ")
        if (
            convert_as_inline
            and el.parent.name not in self.options["keep_inline_images_in"]
        ):
            return alt

        # Remove dataURIs
        if src.startswith("data:") and not self.options["keep_data_uris"]:
            src = src.split(",")[0] + "..."

        return "![%s](%s%s)" % (alt, src, title_part)

    def convert_input(
        self,
        el: Any,
        text: str,
        convert_as_inline: Optional[bool] = False,
        **kwargs,
    ) -> str:
        """Convert checkboxes to Markdown [x]/[ ] syntax."""

        if el.get("type") == "checkbox":
            return "[x] " if el.has_attr("checked") else "[ ] "
        return ""

    def convert_soup(self, soup: Any) -> str:
        return super().convert_soup(soup)  # type: ignore


class _HtmlConverter(object):
    """Anything with content type text/html"""

    def invoke_item(
            self,
            file_stream: BinaryIO,
            encoding='utf-8',
            **kwargs
    ) -> dict:
        soup = BeautifulSoup(file_stream, "html.parser", from_encoding=encoding)
        # Remove javascript and style blocks
        for script in soup(["script", "style"]):
            script.extract()

        # Print only the main content
        body_elm = soup.find("body")
        webpage_text = ""
        try:
            if body_elm:
                webpage_text = _CustomMarkdownify(**kwargs).convert_soup(body_elm)
            else:
                webpage_text = _CustomMarkdownify(**kwargs).convert_soup(soup)
        except RecursionError:
            target = body_elm if body_elm else soup
            webpage_text = target.get_text("\n", strip=True)

        assert isinstance(webpage_text, str)

        # remove leading and trailing \n
        webpage_text = webpage_text.strip()
        output = {
            "content": webpage_text,
            "title": None if soup.title is None else soup.title.string
        }
        return output


class EpubConverter(BaseConverter):
    """
    EPUB 解析器
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        
    def _get_text_from_node(self, dom: Document, tag_name: str) -> str | None:
        """Convenience function to extract a single occurrence of a tag (e.g., title)."""
        texts = self._get_all_texts_from_nodes(dom, tag_name)
        if len(texts) > 0:
            return texts[0]
        else:
            return ""

    def _get_all_texts_from_nodes(self, dom: Document, tag_name: str) -> List[str]:
        """Helper function to extract all occurrences of a tag (e.g., multiple authors)."""
        texts: List[str] = []
        for node in dom.getElementsByTagName(tag_name):
            if node.firstChild and hasattr(node.firstChild, "nodeValue"):
                texts.append(node.firstChild.nodeValue.strip())
        return texts

    def invoke_item(
            self,
            file: str | os.PathLike,
            encoding='utf-8',
            **kwargs
    ) -> list:
        try:
            parse_callback = kwargs.get("parse_callback")
            parse_callback.on_started(**{
                "file": file
            })
            self._html_converter = _HtmlConverter()
            res = []
            with zipfile.ZipFile(file, "r") as z:
                # Locate content.opf
                container_dom = minidom.parse(z.open("META-INF/container.xml"))
                opf_path = container_dom.getElementsByTagName("rootfile")[0].getAttribute(
                    "full-path"
                )

                # Parse content.opf
                opf_dom = minidom.parse(z.open(opf_path))
                metadata: Dict[str, Any] = {
                    "title": self._get_text_from_node(opf_dom, "dc:title"),
                    "authors": self._get_all_texts_from_nodes(opf_dom, "dc:creator"),
                    "language": self._get_text_from_node(opf_dom, "dc:language"),
                    "publisher": self._get_text_from_node(opf_dom, "dc:publisher"),
                    "date": self._get_text_from_node(opf_dom, "dc:date"),
                    "description": self._get_text_from_node(opf_dom, "dc:description"),
                    "identifier": self._get_text_from_node(opf_dom, "dc:identifier"),
                }

                # Extract manifest items (ID → href mapping)
                manifest = {
                    item.getAttribute("id"): item.getAttribute("href")
                    for item in opf_dom.getElementsByTagName("item")
                }

                # Extract spine order (ID refs)
                spine_items = opf_dom.getElementsByTagName("itemref")
                spine_order = [item.getAttribute("idref") for item in spine_items]

                # Convert spine order to actual file paths
                base_path = "/".join(
                    opf_path.split("/")[:-1]
                )  # Get base directory of content.opf
                spine = [
                    f"{base_path}/{manifest[item_id]}" if base_path else manifest[item_id]
                    for item_id in spine_order
                    if item_id in manifest
                ]

                # Extract and convert the content
                epub_idx = 0
                for idx,file_ in enumerate(spine):
                    start_time = time.perf_counter()
                    if file_ in z.namelist():
                        with z.open(file_) as f:
                            html_res = self._html_converter.invoke_item(
                                file_stream = f,
                                encoding=encoding,
                                **kwargs
                            )
                            html_content = html_res.get('content', '').strip()
                            html_content = clean_text_linebreak(html_content)
                            if not html_content:
                                continue
                            epub_idx += 1
                            res.append({
                                "type": "epub",
                                "epub_idx": epub_idx,
                                "content": html_content,
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
            encoding='utf-8',
            **kwargs
    ) -> list:
        res = await asyncio.to_thread(
            self.invoke_item, 
            file, 
            encoding, 
            **kwargs
        )
        return res