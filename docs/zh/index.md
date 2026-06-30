<p align="center">
  <img src="../images/logo.png" style="max-width: 500px; width: 100%;" alt="Logo">
</p>
<a href="https://www.modelscope.cn/models/anyforge/anyparse-models-hub" target="_blank"><img alt="Static Badge" src="https://img.shields.io/badge/%E9%AD%94%E6%90%AD-ModelScope-blue"></a>
<a href="https://huggingface.co/anyforge/anyparse-models-hub" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-blue"></a>
<a href=""><img src="https://img.shields.io/badge/Python->=3.10-aff.svg"></a>

# AnyParse

**AnyParse** 是一个功能强大的多模态文档解析与理解引擎，旨在将复杂文件无缝转换为结构化的 Markdown 和 JSON 格式。无论是基础文本处理、专业文档转换，还是高级视觉语言模型（VLM）和 OCR 识别，AnyParse 都能提供全面的一站式解决方案。

### 核心能力

- **多模态文档理解：** 支持图像与文档的跨模态解析。通过结合 OCR 和 VLM 技术，准确提取非结构化数据。
- **全面的格式覆盖：** 使用单一工具即可轻松解析办公文档、网页、电子表格、电子书和电子邮件。
- **结构化输出：** 将复杂文件转换为标准化的 Markdown 和 JSON，简化下游数据处理和大语言模型（LLM）应用。

### 主要特性

- **文档与版式：** PDF、DOCX、PPTX、XLSX、EPUB、IPYNB
- **文本与标记：** TXT、MD、RST、HTML/XHTML/HTM/SHTML
- **电子表格与数据：** CSV、TSV
- **图像与多媒体：** PNG、JPEG/JPG
- **其他：** EML（电子邮件）
- **内置 CLI、FastAPI**
- **支持在纯 CPU 环境中运行，同时也支持 GPU**
- 按人类阅读顺序输出文本，适用于单栏、多栏和复杂版式
- 保留原始文档结构，包括标题、段落、列表等
- 提取图像、图像描述、表格、表格标题和脚注
- 自动识别文档中的公式并转换为 LaTeX 格式
- 自动识别文档中的表格并转换为 HTML 格式


## 感谢所有贡献者

<a href="https://github.com/anyforge/anyparse/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=anyforge/anyparse" />
</a>


## 许可信息

本仓库采用 [AnyParse 开源许可证](https://github.com/anyforge/anyparse/blob/main/LICENSE.md)，基于 Apache 2.0 并附加了额外条款。


## 致谢

- [MinerU](https://github.com/opendatalab/MinerU)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [GLM-OCR](https://github.com/zai-org/GLM-OCR)
- [pypdfium2](https://github.com/pypdfium2-team/pypdfium2)


## Star 历史

<a href="https://www.star-history.com/?repos=anyforge%2Fanyparse&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=anyforge/anyparse&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=anyforge/anyparse&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=anyforge/anyparse&type=date&legend=top-left" />
 </picture>
</a>
