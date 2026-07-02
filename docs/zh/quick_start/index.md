## 安装

```bash
pip install anyparse-python

# 或者

pip install -e .
```

**anyparse所有模型都是通过 transformers+pytorch 推理, 如果您想使用gpu, 请安装 pytorch+cuda**

## 使用方法

请先下载 `config/config.yaml` 文件到你的项目目录。

### 下载模型

```bash
# 使用 modelscope（默认）
export ANYPARSE_MODEL_MIRROR="modelscope"

# 使用 huggingface
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
res = model.invoke(file="/path/to/your_file")



# 或者异步调用
from anyparse import AsyncAnyParser

model = AsyncAnyParser(config="config/config.yaml")
res = await model.ainvoke(file="/path/to/your_file")
```

### CLI

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
# 启动 fastapi 服务器和 openai 代理
## 使用 restful api 或 openai client 调用
anyparse-cli api --config config/config.yaml --host 0.0.0.0 --port 18007 --seckey 'your_custom_secret_key'
```

- 调用 API

```python
# openai
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:18007/anyparse/openai/v1",
    api_key="your_custom_secret_key",
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
    ],  # data:application/pdf;base64 前缀遵循: client.models.list().data[0].allow_mimetypes
    # extra_body={
    #     "runtimes_args": {
    #         "use_doc_layout": True
    #     }
    # }
)

print(response.choices[0].message.content)




# 或者使用 restful
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
    'file': open(file, 'rb')
}

res = rq.post(url, files=files, data=args, headers=headers)
print(res.json())

```
