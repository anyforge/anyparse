# API

通过 `anyparse-cli api` 启动服务器：请查看 [CLI 工具](./cli_tools.md)

## OpenAI 代理

- 基础 URL: `http://your_ip:your_port/anyparse/openai/v1`
- API Key: `your_custom_secret_key`

### 模型列表和文件 MIME 类型映射

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:18007/anyparse/openai/v1",
    api_key="your_custom_secret_key",
)
print(client.models.list())

```

### 非流式解析文件

```python
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
```

### extra_body 参数

**通过 runtime_args 传递上述调用参数。**

### 输出格式

```json
{
    'code': 2000,
    'data': {
        'metadata': {
            'file_idx': 'task-bb581aed6e4a469694613c454bce72e9',
            'file_md5': '459b4c627b72e50a8e649c6329f1f1a8',
            'file_type': 'pdf',
            'file_name': '11.pdf',
            'file_size': '3.99/KB'},
        'pages': [  # 页面列表
            {
                'id': 1,  # 页面 ID
                'content': 'dsdddddddd',  # 页面内容
                'layout': [
                    {
                        'order_id': 0,  # 阅读顺序
                        'label': 'text',  # 布局类型
                        'box': [578, 1019, 972, 1095], # 布局框
                        'content': 'dsdddddddd'  # 布局内容
                    }
                ],
                'elapse_times': 2.4340291023254395  # 页面耗时
            },
            ...
        ],
        'content': 'dsdddddddd\n我也',  # 文件内容
        'elapse_times': 3.8336679935455322 # 文件耗时
    },
    'msg': 'success'
}

```

### 流式解析文件

```python
import base64

with open("1.pdf", "r", encoding="utf-8") as f:
    text_content = f.read()

encoded_bytes = base64.b64encode(text_content.encode('utf-8'))
base64_str = encoded_bytes.decode('utf-8')

response = client.chat.completions.create(
    model="anyparse",
    stream=True,
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

for x in response:
    print(x)
```

### 创建上传文件

```python
response = client.files.create(
    file=open("11.pdf", "rb"),
    purpose="assistants",
    # extra_body = {
    #     "maxsize": 100.0,
    #     "file_read_size": 1024        
    # }
)
```

### 获取文件信息

```python
client.files.retrieve(
    file_id="your file id",
)
```

### 获取文件内容

```python
response = client.files.content(
    file_id="your file id"
)

with open("your file", "wb") as f:
    for chunk in response.iter_bytes(chunk_size=1024*1024): # 每次读 1MB
        f.write(chunk)

# 详情请查看 headers
print(response.response.headers)
```

### 删除文件

```python
client.files.delete(
    file_id="your file id",
)
```

### 通过文件 ID 解析文件

```python
response = client.chat.completions.create(
    model="anyparse",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "file_id",
                    "file": {
                        "file_data": "your file id"
                    }
                }
            ]
        }
    ],
    stream=False,
    # extra_body = {
    #     "runtimes_args": {
    #         "maxsize": 50
    #     }
    # }
)
```

### 通过 OpenAI responses 接口解析文件

```python
response = client.responses.create(
    model="anyparse",
    input=[
        {
            "role": "user",
            "content": [
                {
                    "type": "file_id",
                    "file": {
                        "file_data": "your file id"
                    }
                }
            ]
        }
    ],
    stream=False,
    # extra_body = {}
)

print(response.output_text)
```


## RESTful API

- 基础 URL: `http://your_ip:your_port/anyparse`
- API Key: `your_custom_secret_key`

### 文件类型

```python
url = 'http://your_ip:your_port/anyparse/filetypes/v1'
headers = {
    "Authorization": f"Bearer your_custom_secret_key",
}
res = httpx.get(url,  headers=headers)

print(res.json())

```


### 解析文件

```python
import httpx
args = {
    "use_doc_cls": False,
    "use_doc_rectifier": False,
    "use_doc_layout": True,
    "stream": False
}
url = f'http://your_ip:your_port/anyparse/invoke/v1'
headers = {
    "Authorization": f"Bearer your_custom_secret_key",
}
files = {
    'file': open('11.pdf','rb')
}

res = httpx.post(url, data=args, headers=headers, files = files)

print(res.json())
```

### API 参数

**通过 args 传递上述调用参数。**

### 输出格式

```json
{
    'code': 2000,
    'data': {
        'metadata': {
            'file_idx': 'task-bb581aed6e4a469694613c454bce72e9',
            'file_md5': '459b4c627b72e50a8e649c6329f1f1a8',
            'file_type': 'pdf',
            'file_name': '11.pdf',
            'file_size': '3.99/KB'},
        'pages': [  # 页面列表
            {
                'id': 1,  # 页面 ID
                'content': 'dsdddddddd',  # 页面内容
                'layout': [
                    {
                        'order_id': 0,  # 阅读顺序
                        'label': 'text',  # 布局类型
                        'box': [578, 1019, 972, 1095], # 布局框
                        'content': 'dsdddddddd'  # 布局内容
                    }
                ],
                'elapse_times': 2.4340291023254395  # 页面耗时
            },
            ...
        ],
        'content': 'dsdddddddd\n我也',  # 文件内容
        'elapse_times': 3.8336679935455322 # 文件耗时
    },
    'msg': 'success'
}

```


### 流式解析文件

```python
import json
import httpx

args = {
    "use_doc_cls": False,
    "use_doc_rectifier": False,
    "use_doc_layout": True,
    "stream": True
}
url = f'http://your_ip:your_port/anyparse/invoke/v1'
headers = {
    "Authorization": f"Bearer your_custom_secret_key",
}
files = {
    'file': open('11.pdf','rb')
}
with httpx.Client(timeout=3600) as client:
    with client.stream("POST", url, data=args, headers=headers, files = files) as response:
        response.raise_for_status()
        buffer = ""
        for chunk_bytes in response.iter_bytes():
            buffer += chunk_bytes.decode("utf-8")
            lines = buffer.split("\n")
            buffer = lines[-1]  # 保留不完整行

            for line in lines[:-1]:
                line = line.strip()
                if line and line.startswith("data:"):
                    payload = line[5:].strip()
                    if not payload:
                        continue
                    data = json.loads(payload)
                    text = data.get("content", "")
                    pageid = int(data.get("id", 1)) + 1
                    print(data)
```
