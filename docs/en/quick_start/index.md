## Installation

```bash
pip install anyparse-python

# or

pip install -e .
```

**anyparse all models infer by transformers+pytorch, if you want to use gpu, please install pytorch+cuda**

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