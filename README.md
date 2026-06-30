<p align="center">
  <img src="./docs/images/logo.png" style="width: 500px;" alt="Logo">
</p>
<a href="https://www.modelscope.cn/models/anyforge/anyparse-models-hub" target="_blank"><img alt="Static Badge" src="https://img.shields.io/badge/%E9%AD%94%E6%90%AD-ModelScope-blue"></a>
<a href="https://huggingface.co/anyforge/anyparse-models-hub" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-blue"></a>
<a href=""><img src="https://img.shields.io/badge/Python->=3.10-aff.svg"></a>

[English](README.md) | [简体中文](README_zh.md)

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


# Insduction

## Installation

```bash
pip install anyparse-python

# or

pip install -e .
```

## Usage

please download `config/config.yaml` into your project directory.

### Download Models

```bash
# use modelscope (default)
export ANYPARSE_MODEL_MIRROR="modelscope"

# use huggingface
export ANYPARSE_MODEL_MIRROR="huggingface"

# download models
anyparse-cli download --config config/config.yaml --model
```

### Models Hub
- [AnyParse Models Hub ModelScope](https://www.modelscope.cn/models/anyforge/anyparse-models-hub)
- [AnyParse Models Hub HuggingFace](https://huggingface.co/anyforge/anyparse-models-hub)


### Python

```python
# Sync
from anyparse import AnyParser

model = AnyParser(config="config/config.yaml")
res = model.invoke(file = "/path/to/your_file")



# or Async
from anyparse import AsyncAnyParser

model = AsyncAnyParser(config="config/config.yaml")
res = await model.ainvoke(file = "/path/to/your_file")
```

### CLI

```bash
# help

anyparse-cli --help

# parse file
anyparse-cli parse --config config/config.yaml --file /path/to/your_file

# start api server
anyparse-cli api --config config/config.yaml

# see allowed file types
anyparse-cli allow --config config/config.yaml

# see commands help
anyparse-cli [COMMAND] --help
```

### API

- start api server

```bash
# start fastapi server and openai proxy
## use restful api or openai client call
anyparse-cli api --config config/config.yaml --host 0.0.0.0 --port 18007 --seckey 'your_custom_secret_key'
```

- call api

```python
# openai
from openai import OpenAI

client = OpenAI(
    base_url = "http://localhost:18007/anyparse/openai/v1",
    api_key = "your_custom_secret_key",
)
## get model id and allowed file types
print(client.models.list())



## parse file
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
    ],  # data:application/pdf;base64 prefix follow: client.models.list().data[0].allow_mimetypes
    # extra_body={
    #     "runtimes_args": {
    #         "use_doc_layout": True
    #     }
    # }
)

print(response.choices[0].message.content)



# or restful
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

**Details and Documentation see [docs](https://anyforge.github.io/anyparse/)**

## TODO LIST

- audio transcription
- video transcription

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
