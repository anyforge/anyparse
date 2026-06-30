# cli tools

## help

```bash
anyparse-cli --help
```

## help command

```bash
anyparse-cli [COMMAND] --help
```

## parse file
```bash
anyparse-cli parse --config config/config.yaml --file /path/to/your_file

```
**params:**

- config: config file path
- file: file path
- log_level: log level


## start api server and openai proxy
```bash
anyparse-cli api --config config/config.yaml

```

**params:**

- config: config file path
- host: api server host, default: 0.0.0.0
- port: api server port, default: 18007
- prefix: api prefix, default: /anyparse
- seckey: api secret key, default: config.yaml.auths.secret_key
- api_extra_args: json string, extra args for fastapi, default: ""


## allowed file types
```bash
anyparse-cli allow --config config/config.yaml
```

## download models
```bash
anyparse-cli download --config config/config.yaml --model
```