<p align="center">
  <img src="./images/logo.png" style="max-width: 500px; width: 100%;" alt="Logo">
</p>

<p align="center">
    <a href="https://www.modelscope.cn/models/anyforge/anyparse-models-hub" target="_blank"><img alt="Static Badge" src="https://img.shields.io/badge/%E9%AD%94%E6%90%AD-ModelScope-blue"></a>
    <a href="https://huggingface.co/anyforge/anyparse-models-hub" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-blue"></a>
    <a href=""><img src="https://img.shields.io/badge/Python->=3.10-aff.svg"></a>
    <a href="https://pypi.org/project/anyparse-python/"><img src="https://img.shields.io/pypi/v/anyparse-python"></a>
    <a href="https://pypi.org/project/anyparse-python/"><img src="https://img.shields.io/pypi/pyversions/anyparse-python"></a>
    <a href="https://pypi.org/project/anyparse-python/"><img src="https://static.pepy.tech/badge/anyparse-python"></a>
</p>

# AnyParse

**AnyParse** is a powerful multimodal document parsing and understanding engine designed to seamlessly convert complex files into structured Markdown and JSON formats. Whether it's basic text processing, professional document conversion, or advanced Vision-Language Models (VLM) and OCR recognition, AnyParse provides a comprehensive, one-stop solution.

### Core Capabilities

- **Multimodal Document Understanding:** Supports cross-modal parsing of images and documents. By combining OCR and VLM technologies, it accurately extracts unstructured data.
- **Comprehensive Format Coverage:** Easily parses office documents, web pages, spreadsheets, e-books, and emails with a single tool.
- **Structured Output:** Transforms complex files into standardized Markdown and JSON, streamlining downstream data processing and Large Language Model (LLM) applications.

### Key Features

- **Documents & Layouts:** PDF, DOCX, PPTX, XLSX, EPUB, IPYNB
- **Text & Markup:** TXT, MD, RST, HTML/XHTML/HTM/SHTML
- **Spreadsheets & Data:** CSV, TSV
- **Images & Multimedia:** PNG, JPEG/JPG
- **Others:** EML (Emails)
- **Built-in CLI, FastAPI**
- **Supports running in a pure CPU environment, and also supports GPU**
- Output text in human reading order, suitable for single-column, multi-column and complex layouts
- Retain the original document structure, including titles, paragraphs, lists, etc.
- Extract images, image descriptions, tables, table titles and footnotes
- Automatically identify and convert formulas in documents to LaTeX format
- Automatically identify and convert tables in documents to HTML format

### resources

- **docs: [AnyParse docs](https://anyforge.github.io/anyparse)**
- **pypi: [anyparse-python](https://pypi.org/project/anyparse-python/)**
- **[ModelScope Skills](https://www.modelscope.cn/skills/anyforge/anyparse-skill)**
- **[SkillHub](https://skillhub.cn/skills/anyparse-skill)**
- **[ClawHub](https://clawhub.ai/anyforge/skills/anyparse-skill)**

## All Thanks To Our Contributors

<a href="https://github.com/anyforge/anyparse/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=anyforge/anyparse" />
</a>


## License Information

This repository is licensed under the [AnyParse Open Source License](https://github.com/anyforge/anyparse/blob/main/LICENSE.md), based on Apache 2.0 with additional conditions.


## Acknowledgments

- [MinerU](https://github.com/opendatalab/MinerU)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [GLM-OCR](https://github.com/zai-org/GLM-OCR)
- [pypdfium2](https://github.com/pypdfium2-team/pypdfium2)


## Star History

<a href="https://www.star-history.com/?repos=anyforge%2Fanyparse&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=anyforge/anyparse&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=anyforge/anyparse&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=anyforge/anyparse&type=date&legend=top-left" />
 </picture>
</a>
