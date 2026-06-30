# 模型源

## 模型仓库

- [AnyParse 模型仓库 ModelScope](https://www.modelscope.cn/models/anyforge/anyparse-models-hub)
- [AnyParse 模型仓库 HuggingFace](https://huggingface.co/anyforge/anyparse-models-hub)


## 下载模型

```bash
# 使用 modelscope（默认）
export ANYPARSE_MODEL_MIRROR="modelscope"

# 使用 huggingface
export ANYPARSE_MODEL_MIRROR="huggingface"

# 下载模型
anyparse-cli download --config config/config.yaml --model
```

## 注意事项

你可以自行下载模型，并修改 `config/config.yaml` 文件以使用你自己的模型。
