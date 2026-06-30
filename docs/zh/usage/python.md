# Python API

## 安装

```bash
pip install anyparse-python

# 或者

pip install -e .
```

## 同步 API
```python

from anyparse import AnyParser

model = AnyParser(config="config/config.yaml")
res = model.invoke(file = "/path/to/your_file")

```


## 异步 API

```python
from anyparse import AsyncAnyParser

model = AsyncAnyParser(config="config/config.yaml")
res = await model.ainvoke(file = "/path/to/your_file")

```

## config.yaml

```yaml
product_name: "anyparse"

model_id: "${product_name}"

modelapi:
    prefix: "/${product_name}"
    host: ${oc.env:api_host, "0.0.0.0"}
    port: ${oc.env:api_port, "18007"}
    path:
        # restful api
        filetypes: "/filetypes/v1"
        invoke: "/invoke/v1"
        # openai proxy
        openai_model_list: "/openai/v1/models"
        openai_create_file: "/openai/v1/files"
        openai_retrieve_file: "/openai/v1/files"
        openai_content_file: "/openai/v1/files"
        openai_delete_file: "/openai/v1/files"
        openai_chat_completions: "/openai/v1/chat/completions"
        openai_responses: "/openai/v1/responses"

auths:
    # api key
    secret_key: ${oc.env:api_secret_key, "sk_6c5...f7bf"}

filetypes: 
    text: ['txt']
    html: ['htm','html','xhtml','shtml']
    markdown: ['md','rst']
    image: ['png', 'jpeg', 'jpg']
    pdf: ['pdf']
    epub: ['epub']
    office: ['docx','pptx','xlsx']
    csv: ['tsv','csv']
    ipynb: ['ipynb']
    email: ['eml']

mimetypes:
    txt: "data:text/plain;base64,"
    htm: "data:text/html;base64,"
    html: "data:text/html;base64,"
    xhtml: "data:text/html;base64,"
    shtml: "data:text/html;base64,"
    md: "data:text/markdown;base64,"
    rst: "data:text/rst;base64,"
    png: "data:image/png;base64,"
    jpeg: "data:image/jpeg;base64,"
    jpg: "data:image/jpeg;base64,"
    webp: "data:image/webp;base64,"
    pdf: "data:application/pdf;base64,"
    epub: "data:application/epub;base64,"
    docx: "data:application/docx;base64,"
    pptx: "data:application/pptx;base64,"
    xlsx: "data:application/xlsx;base64,"
    tsv: "data:text/csv;base64,"
    csv: "data:text/csv;base64,"
    ipynb: "data:application/ipynb;base64,"
    eml: "data:message/eml;base64,"
    wav: "data:audio/wav;base64,"
    mp3: "data:audio/mpeg;base64,"
    aac: "data:audio/aac;base64,"
    flac: "data:audio/flac;base64,"
    mp4: "data:video/mp4;base64,"
    mov: "data:video/mov;base64,"
    mkv: "data:video/mkv;base64,"
    webm: "data:video/webm;base64,"
    avi: "data:video/avi;base64,"

anyparse:
    # 缓存文件保存目录
    cache_dir: "~/.cache/anyparse"
    # 计算文件 MD5
    autocal_md5: false
    # 检测编码
    autodetect_encoding: true
    # 图片和 PDF
    ## 使用文档方向分类
    use_doc_cls: false
    ## 使用文档矫正
    use_doc_rectifier: false
    ## 使用文档布局
    use_doc_layout: true
    ## 文档布局识别的最小图片尺寸
    doc_layout_image_min_size: 500
    ## 使用图片缩放
    use_image_resize: false
    ## 图片批处理大小
    image_batch_size: 1
    ## OCR 批处理大小
    ocr_batch_size: 1

    # 文本编码
    text_encoding: "utf-8"
    # csv tsv 分块大小
    table_chunk_size: null
    # PDF 页面转图片的 DPI
    dpi: 200
    # 解析详细输出
    verbose: true
    # 解析流式处理
    stream: false
    
    ### docx
    docx_extract_headers_footers: true
    docx_extract_images: true

    ### pptx
    pptx_extract_images: true

    ### xlsx
    excel_max_rows: null  # 未使用
    excel_extract_images: true

    ### 文档方向分类，请参考 paddleocr pp_lcnet_x1_0_doc_ori
    doc_cls:
        batch_size: 1
        model_path: "~/.cache/anyparse/models/pp_lcnet_x1_0_doc_ori"
        dtype: "auto"
        device_map: "auto"

    ## 文档矫正，请参考 paddleocr uvdoc
    doc_rectifier:
        batch_size: 1
        model_path: "~/.cache/anyparse/models/pp_uvdoc"
        dtype: "auto"
        device_map: "auto"

    ### 文档布局，请参考 paddleocr ppdoclayout-v3
    layout:
        model_path: "~/.cache/anyparse/models/ppdoclayout-v3"
        threshold: 0.3
        batch_size: 1
        dtype: "auto"
        device_map: "auto"
        layout_nms: true

    vlm:
        model_type: "paddleocrvl" # paddleocr, glmocr_v1, paddleocrvl, vllm
        
        paddleocr:
            model_class: "PaddleOCRClient.PPOCRV6"
            model_path: "~/.cache/anyparse/models/paddleocrv6-small"
            batch_size: 6
            dtype: 'auto'
            device_map: 'auto'

        glmocr_v1:
            model_class: "GlmOCRClient.GLMOCRV1"
            model_path: "~/.cache/anyparse/models/glmocr-v1"
            batch_size: 1
            max_new_tokens: 8192
            dtype: "auto"
            device_map: "auto"
            min_pixels: 12544 # 112 * 112
            max_pixels: 71372800 # 14 * 14 * 4 * 1280  

        paddleocrvl:
            model_class: "PaddleOCRVLClient.PPOCRVLClient"
            model_path: "~/.cache/anyparse/models/paddleocrvl-v1.6"
            batch_size: 1,
            max_new_tokens: 16384
            dtype: "auto"
            device_map: "auto"  
            attn_implementation: null  
            truncate_content: true
            truncate_content_list: [5000, 50] 
        
        vllm: # paddlocrvl-1.6 openai server
            model_class: "OpenAIClient.OpenAIClient"
            base_url: "http://localhost:18003/v1"
            api_key: "sk-123456"
            model: "PaddleOCR-VL-1.6"
            stream: false
            timeout: 1800.0
            max_retries: 2
            batch_size: 8
            max_new_tokens: 8192
            # paddleocrvl 提示词映射
            task_prompt_map:
                abstract: "OCR:"
                algorithm: "OCR:"
                content: "OCR:"
                doc_title: "OCR:"
                figure_title: "OCR:"
                paragraph_title: "OCR:"
                reference_content: "OCR:"
                text: "OCR:"
                vertical_text: "OCR:"
                vision_footnote: "OCR:"
                seal: "OCR:"
                formula_number: "OCR:"
                header: "OCR:"
                footer: "OCR:"
                number: "OCR:"
                footnote: "OCR:"
                aside_text: "OCR:"
                reference: "OCR:"
                footer_image: "OCR:"
                header_image: "OCR:"
                image: "OCR:"
                table: "Table Recognition:"
                display_formula: "Formula Recognition:"
                inline_formula: "Formula Recognition:"
                chart: "Chart Recognition:"
            client_args:
            call_args:
            prompt_template: >-
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url", 
                                "image_url": {
                                    "url": "{{ data_url }}"
                                }
                            },
                            {
                                "type": "text", 
                                "text": "{{ prompt }}"
                            }
                        ]
                    }
                ]

    # converters_models:
    #     - name: "markdown"
    #       model_class: "mkd.MkdConverter"
        
logger: # logoru 配置
  type: "loguru"
  filename: ./logs/${product_name}.log
  level: "DEBUG"
  loguru:
      encoding: "utf-8"
      mode: "a+"
      rotation: "00:00"
      retention: "30 days"
      colorize: false
      enqueue: true
      backtrace: true
      diagnose: true
      compression: null
      strformat: "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{process}</cyan>:<cyan>{thread}</cyan> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>"

```

## 注意事项

`${oc.env:api_host, "0.0.0.0"}` 类似于 omegaconf 语法，可以通过环境变量修改：`export api_host=0.0.0.0` 或者 `source config/config.env`


## invoke 和 ainvoke 参数

invoke 和 ainvoke 的实时参数，例如：

```python

model.invoke(
    file = "/path/to/your_file",
    autocal_md5 = True,
    ...
)

```

- **autocal_md5**: 是否自动计算 MD5。默认值：`True`。
- **autodetect_encoding**: 是否自动检测编码。默认值：`True`。
- **use_doc_cls**: 是否执行文档方向矫正。默认值：`False`。
- **use_doc_rectifier**: 是否执行文档矫正。默认值：`False`。
- **use_doc_layout**: 是否执行文档布局识别。默认值：`True`。
- **doc_layout_image_min_size**: 文档布局识别的最小图片尺寸。默认值：`400`。
- **use_image_resize**: 是否缩放图片。默认值：`False`。
- **text_encoding**: 文本编码格式。默认值：`"utf-8"`。
- **table_chunk_size**: 表格分块大小。默认值：`None`。
- **table_custom_separator**: 表格自定义分隔符。默认值：`None`。
- **table_sniffer**: 是否自动检测表格分隔符。默认值：`True`。
- **dpi**: 图片分辨率（DPI）。默认值：`200`。
- **verbose**: 是否启用详细输出。默认值：`True`。
- **stream**: 是否使用流式处理。默认值：`False`。
- **docx_extract_headers_footers**: 是否从 DOCX 文件中提取页眉和页脚。默认值：`True`。
- **docx_extract_images**: 是否从 DOCX 文件中提取图片。默认值：`True`。
- **pptx_extract_images**: 是否从 PPTX 文件中提取图片。默认值：`True`。
- **excel_extract_images**: 是否从 Excel 文件中提取图片。默认值：`True`。
- **excel_max_rows**: 从 Excel 文件中提取的最大行数。默认值：`None`。
- **max_new_tokens**: 最大生成长度。默认值：`8192`。
- **image_batch_size**: 图片处理批大小。默认值：`1`。
- **ocr_batch_size**: 文本（OCR）处理批大小。默认值：`1`。
