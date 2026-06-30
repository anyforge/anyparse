# CLI 工具

## 帮助

```bash
anyparse-cli --help
```

## 命令帮助

```bash
anyparse-cli [COMMAND] --help
```

## 解析文件

```bash
anyparse-cli parse --config config/config.yaml --file /path/to/your_file

```

**参数说明：**

- config: 配置文件路径
- file: 文件路径
- log_level: 日志级别


## 启动 API 服务器和 OpenAI 代理

```bash
anyparse-cli api --config config/config.yaml

```

**参数说明：**

- config: 配置文件路径
- host: API 服务器主机，默认：0.0.0.0
- port: API 服务器端口，默认：18007
- prefix: API 前缀，默认：/anyparse
- seckey: API 密钥，默认：config.yaml.auths.secret_key
- api_extra_args: JSON 字符串，FastAPI 额外参数，默认：""


## 查看支持的文件类型

```bash
anyparse-cli allow --config config/config.yaml
```

## 下载模型

```bash
anyparse-cli download --config config/config.yaml --model
```
