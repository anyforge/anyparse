```
 █████╗ ███╗   ██╗██╗   ██╗██████╗  █████╗ ██████╗ ███████╗███████╗
██╔══██╗████╗  ██║╚██╗ ██╔╝██╔══██╗██╔══██╗██╔══██╗██╔════╝██╔════╝
███████║██╔██╗ ██║ ╚████╔╝ ██████╔╝███████║██████╔╝███████╗█████╗  
██╔══██║██║╚██╗██║  ╚██╔╝  ██╔═══╝ ██╔══██║██╔══██╗╚════██║██╔══╝  
██║  ██║██║ ╚████║   ██║   ██║     ██║  ██║██║  ██║███████║███████╗
╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝
```

<a href="https://www.modelscope.cn/models/anyforge/anyparse-models-hub" target="_blank"><img alt="Static Badge" src="https://img.shields.io/badge/%E9%AD%94%E6%90%AD-ModelScope-blue"></a>
<a href="https://huggingface.co/anyforge/anyparse-models-hub" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-blue"></a>
<a href=""><img src="https://img.shields.io/badge/Python->=3.10-aff.svg"></a>

[English](README.md) | [简体中文](README_zh.md)

# AnyParse

**AnyParse** 是一个功能强大的多模态文档解析与理解引擎，旨在将复杂文件无缝转换为结构化的 Markdown 和 JSON 格式。无论是基础文本处理、专业文档转换，还是先进的视觉语言模型（VLM）和 OCR 识别，AnyParse 都能提供全面的一站式解决方案。

### 核心能力

- **多模态文档理解：** 支持图像与文档的跨模态解析，通过结合 OCR 和 VLM 技术，精准提取非结构化数据。
- **全面格式覆盖：** 单一工具即可轻松解析办公文档、网页、电子表格、电子书和邮件。
- **结构化输出：** 将复杂文件转换为标准化的 Markdown 和 JSON，简化下游数据处理和大语言模型（LLM）应用流程。

### 主要特性

- **文档与布局：** PDF、DOCX、PPTX、XLSX、EPUB、IPYNB
- **文本与标记：** TXT、MD、RST、HTML/XHTML/HTM/SHTML
- **电子表格与数据：** CSV、TSV
- **图像与多媒体：** PNG、JPEG/JPG
- **其他：** EML（邮件）
- **内置 CLI、FastAPI**
- **支持纯 CPU 环境运行，也支持 GPU**
- 按人类阅读顺序输出文本，适用于单栏、多栏和复杂布局
- 保留原始文档结构，包括标题、段落、列表等
- 提取图像、图像描述、表格、表格标题和脚注
- 自动识别文档中的公式并转换为 LaTeX 格式
- 自动识别文档中的表格并转换为 HTML 格式


# 介绍

## 安装

```bash
pip install anyparse-python

# 或者

pip install -e .
```

## 使用方法

请将 `config/config.yaml` 下载到您的项目目录中。

### 下载模型

```bash
# 使用 ModelScope（默认）
export ANYPARSE_MODEL_MIRROR="modelscope"

# 使用 HuggingFace
export ANYPARSE_MODEL_MIRROR="huggingface"

# 下载模型
anyparse-cli download --config config/config.yaml --model
```

### 模型仓库

- [AnyParse 模型仓库 ModelScope](https://www.modelscope.cn/models/anyforge/anyparse-models-hub)
- [AnyParse 模型仓库 HuggingFace](https://huggingface.co/anyforge/anyparse-models-hub)


### Python

```python
# 同步调用
from anyparse import AnyParser

model = AnyParser(config="config/config.yaml")
res = model.invoke(file = "/path/to/your_file")


# 或者异步调用
from anyparse import AsyncAnyParser

model = AsyncAnyParser(config="config/config.yaml")
res = await model.ainvoke(file = "/path/to/your_file")
```

### CLI 命令行

```bash
# 查看帮助
anyparse-cli --help

# 解析文件
anyparse-cli parse --config config/config.yaml --file /path/to/your_file

# 启动 API 服务器
anyparse-cli api --config config/config.yaml

# 查看支持的文件类型
anyparse-cli allow --config config/config.yaml

# 查看命令帮助
anyparse-cli [COMMAND] --help
```

### API

- 启动 API 服务器

```bash
# 启动 FastAPI 服务器和 OpenAI 代理
## 使用 RESTful API 或 OpenAI 客户端调用
anyparse-cli api --config config/config.yaml --host 0.0.0.0 --port 18007 --seckey 'your_custom_secret_key'
```

- 调用 API

```python
# OpenAI 方式
from openai import OpenAI

client = OpenAI(
    base_url = "http://localhost:18007/anyparse/openai/v1",
    api_key = "your_custom_secret_key",
)
## 获取模型 ID 和支持的文件类型
print(client.models.list())


## 解析文件
import base64

with open("1.pdf", "r", encoding="utf-8") as f:
    text_content = f.read()

encoded_bytes = base64.b64encode(text_content.encode('utf-8'))
base64_str = encoded_bytes.decode('utf-8')

response = client.chat.completions.create(
    model="anyparse",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "file",
                    "file": {
                        "file_data": f"data:application/pdf;base64,{base64_str}"
                    }
                }
            ]
        }
    ],  # data:application/pdf;base64 前缀参考: client.models.list().data[0].allow_mimetypes
    # extra_body={
    #     "runtimes_args": {
    #         "use_doc_layout": True
    #     }
    # }
)

print(response.choices[0].message.content)


# 或者 RESTful 方式
import requests as rq

headers = {
    "Authorization": "Bearer your_custom_secret_key"
}

url = "http://localhost:18007/anyparse/invoke/v1"

args = {
    "use_doc_cls": False,
    "use_doc_rectifier": False,
    "use_doc_layout": True
}

file = '/path/to/your_file'

files = {
    'file': open(file,'rb')
}

res = rq.post(url, files = files, data = args, headers = headers)
print(res.json())
```

**详细信息和文档请参见 [docs](https://anyforge.github.io/anyparse/)**

## TODO LIST

- audio transcription
- video transcription

## 感谢所有贡献者

<a href="https://github.com/anyforge/anyparse/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=anyforge/anyparse" />
</a>


## 许可证信息

本仓库基于 Apache 2.0 许可证，并附加特定条款，完整许可证请参见 [AnyParse 开源许可证](https://github.com/anyforge/anyparse/blob/main/LICENSE.md)。


## 致谢

- [MinerU](https://github.com/opendatalab/MinerU)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [GLM-OCR](https://github.com/zai-org/GLM-OCR)
- [pypdfium2](https://github.com/pypdfium2-team/pypdfium2)


## Star 增长曲线

<a href="https://www.star-history.com/?repos=anyforge%2Fanyparse&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=anyforge/anyparse&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=anyforge/anyparse&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=anyforge/anyparse&type=date&legend=top-left" />
 </picture>
</a>
