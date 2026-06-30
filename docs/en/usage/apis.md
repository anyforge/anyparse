# API

start server by `anyparse-cli api`: see [cli tools](./cli_tools.md)

## openai proxy

- base url: `http://your_ip:your_port/anyparse/openai/v1`
- api key: `your_custom_secret_key`


### models list and file mimetypes mapping
```python
from openai import OpenAI

client = OpenAI(
    base_url = "http://localhost:18007/anyparse/openai/v1",
    api_key = "your_custom_secret_key",
)
print(client.models.list())

```

### parse file not stream
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
    ],  # data:application/pdf;base64 prefix follow: client.models.list().data[0].allow_mimetypes
    # extra_body={
    #     "runtimes_args": {
    #         "use_doc_layout": True
    #     }
    # }
)

print(response.choices[0].message.content)
```

### extra_body

**Pass the above invoke parameters via runtime_args.**

### output format
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
        'pages': [  # page list
            {
                'id': 1,  # page id
                'content': 'dsdddddddd',  # page content
                'layout': [
                    {
                        'order_id': 0,  # reading order 
                        'label': 'text',  # layout type
                        'box': [578, 1019, 972, 1095], # layout box
                        'content': 'dsdddddddd'  # layout content
                    }
                ],
                'elapse_times': 2.4340291023254395  # page elapse time
            },
            ...
        ],
        'content': 'dsdddddddd\n我也',  # file content
        'elapse_times': 3.8336679935455322 # file elapse time
    },
    'msg': 'success'
}

```

### parse file stream
```python
import base64

with open("1.pdf", "r", encoding="utf-8") as f:
    text_content = f.read()

encoded_bytes = base64.b64encode(text_content.encode('utf-8'))
base64_str = encoded_bytes.decode('utf-8')

response = client.chat.completions.create(
    model="anyparse",
    stream = True,
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

for x in response:
    print(x)
```

### create upload file
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

### retrieve file
```python
client.files.retrieve(
    file_id = "your file id",
)
```

### retrieve file content
```python
response = client.files.content(
    file_id = "your file id"
)

with open("your file", "wb") as f:
    for chunk in response.iter_bytes(chunk_size=1024*1024): # 每次读 1MB
        f.write(chunk)

# details see headers
print(response.response.headers)
```

### delete file
```python
client.files.delete(
    file_id = "your file id",
)
```

### parse file by file id
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
    stream = False,
    # extra_body = {
    #     "runtimes_args": {
    #         "maxsize": 50
    #     }
    # }
)
```

### parse file by openai responses
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
    stream = False,
    # extra_body = {}
)

print(response.output_text)
```


## restful api

- base url: `http://your_ip:your_port/anyparse`
- api key: `your_custom_secret_key`


### file types
```python
url = 'http://your_ip:your_port/anyparse/filetypes/v1'
headers = {
    "Authorization": f"Bearer your_custom_secret_key",
}
res = httpx.get(url,  headers=headers)

print(res.json())

```


### parse file

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

### api args

**Pass the above invoke parameters via args.**

### output format
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
        'pages': [  # page list
            {
                'id': 1,  # page id
                'content': 'dsdddddddd',  # page content
                'layout': [
                    {
                        'order_id': 0,  # reading order 
                        'label': 'text',  # layout type
                        'box': [578, 1019, 972, 1095], # layout box
                        'content': 'dsdddddddd'  # layout content
                    }
                ],
                'elapse_times': 2.4340291023254395  # page elapse time
            },
            ...
        ],
        'content': 'dsdddddddd\n我也',  # file content
        'elapse_times': 3.8336679935455322 # file elapse time
    },
    'msg': 'success'
}

```


### parse file stream

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