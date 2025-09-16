# AnyParse
> any file parse to markdown(open source for now: pdf, image, office, html, textbase, more in the future)
> This is base anyparse, we are training plus.

<a href="https://huggingface.co/anyforge/anyparse" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-blue"></a>
<a href="https://www.modelscope.cn/models/anyforge/anyparse" target="_blank"><img alt="Static Badge" src="https://img.shields.io/badge/%E9%AD%94%E6%90%AD-ModelScope-blue"></a>
<a href=""><img src="https://img.shields.io/badge/Python->=3.10-aff.svg"></a>
<a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>

```
    _                ____
   / \   _ __  _   _|  _ \ __ _ _ __ ___  ___
  / _ \ | '_ \| | | | |_) / _` | '__/ __|/ _ \
 / ___ \| | | | |_| |  __/ (_| | |  \__ \  __/
/_/   \_\_| |_|\__, |_|   \__,_|_|  |___/\___|
               |___/

```

- Github：[AnyParse](https://github.com/anyforge/anyparse)
- Hugging Face: [AnyParse](https://huggingface.co/anyforge/anyparse)
- ModelScope: [AnyParse](https://www.modelscope.cn/models/anyforge/anyparse)
- if need doc layout detect: [anydoclayout](https://github.com/anyforge/anydoclayout)
- if need doc table detect: [anytable](https://github.com/anyforge/anytable)

## 1. Usage

- download models to "./resource/models": by ModelScope or Hugging Face

```python
from anyparse.parser import AnyParse
from anyparse.settings import Settings

args = Settings().model_dump() ## see Settings configs
model = AnyParse(args)

file = '1.pdf'

res = model.invoke(file,ocr_mode = "base", stream = False)
res = model.invoke(file,ocr_mode = "plus", stream = False)
```

## 2. TodoList
- audio file parse
- video file parse
- we are training anyocr-vlm by 10Mdocuments

### Business cooperation or get a more powerful version
- email: christnowx@qq.com

### Buy me a coffee

- 微信(WeChat)

<div align="left">
    <img src="./zanshan.jpg" width="30%" height="30%">
</div>

### Star History

[![Star History Chart](https://api.star-history.com/svg?repos=anyforge/anyparse&type=Date)](https://www.star-history.com/#anyforge/anyparse&Date)

### Thanks
- RapidOCR
- Nanonets-OCR-s
