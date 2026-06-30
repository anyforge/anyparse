# Model Source

## Models Hub

- [AnyParse Models Hub ModelScope](https://www.modelscope.cn/models/anyforge/anyparse-models-hub)
- [AnyParse Models Hub HuggingFace](https://huggingface.co/anyforge/anyparse-models-hub)


## Download Models

```bash
# use modelscope (default)
export ANYPARSE_MODEL_MIRROR="modelscope"

# use huggingface
export ANYPARSE_MODEL_MIRROR="huggingface"

# download models
anyparse-cli download --config config/config.yaml --model
```

## Notes

you can download models by yourself, and modify the `config/config.yaml` file to use your own models.
