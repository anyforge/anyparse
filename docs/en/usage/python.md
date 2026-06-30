# Python API

## Installation

```bash
pip install anyparse-python

# or

pip install -e .
```

## Sync API
```python

from anyparse import AnyParser

model = AnyParser(config="config/config.yaml")
res = model.invoke(file = "/path/to/your_file")

```


## Async API

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
    secret_key: ${oc.env:api_secret_key, "sk_6c5aa04e523de79518620095a0e2f7bf"}

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
    # cache to save file
    cache_dir: "~/.cache/anyparse"
    # cal file md5
    autocal_md5: false
    # detect encoding
    autodetect_encoding: true
    # image and pdf
    ## use doc ori cls
    use_doc_cls: false
    ## use doc rectifier
    use_doc_rectifier: false
    ## use doc layout
    use_doc_layout: true
    ## doc layout min size
    doc_layout_image_min_size: 500
    ## use image resize
    use_image_resize: false
    ## image batch size
    image_batch_size: 1
    ## ocr batch size
    ocr_batch_size: 1

    #  text encoding
    text_encoding: "utf-8"
    # csv tsv chunk size
    table_chunk_size: null
    # pdf page to image dpi
    dpi: 200
    # parse verbose
    verbose: true
    # parse stream
    stream: false
    
    ### docx
    docx_extract_headers_footers: true
    docx_extract_images: true

    ### pptx
    pptx_extract_images: true

    ### xlsx
    excel_max_rows: null  # unused
    excel_extract_images: true

    ### doc ori cls see paddleocr pp_lcnet_x1_0_doc_ori
    doc_cls:
        batch_size: 1
        model_path: "~/.cache/anyparse/models/pp_lcnet_x1_0_doc_ori"
        dtype: "auto"
        device_map: "auto"

    ## uvdoc see paddleocr uvdoc
    doc_rectifier:
        batch_size: 1
        model_path: "~/.cache/anyparse/models/pp_uvdoc"
        dtype: "auto"
        device_map: "auto"

    ### doc layout see paddleocr ppdoclayout-v3
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
            # paddleocrvl prompt mapping
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
        
logger: # logoru config
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

## notice

`${oc.env:api_host, "0.0.0.0"}` like omegaconf can be modified by environment variables: `export api_host=0.0.0.0` or `source config/config.env`


## invoke and ainvoke params

realtime args for invoke and ainvoke,like:

```python

model.invoke(
    file = "/path/to/your_file",
    autocal_md5 = True,
    ...
)

```

- **autocal_md5**: Whether to automatically calculate MD5. Default: `True`.
- **autodetect_encoding**: Whether to automatically detect encoding. Default: `True`.
- **use_doc_cls**: Whether to perform document orientation correction. Default: `False`.
- **use_doc_rectifier**: Whether to perform document rectification. Default: `False`.
- **use_doc_layout**: Whether to perform document layout recognition. Default: `True`.
- **doc_layout_image_min_size**: Minimum image size for document layout recognition. Default: `400`.
- **use_image_resize**: Whether to resize images. Default: `False`.
- **text_encoding**: Text encoding format. Default: `"utf-8"`.
- **table_chunk_size**: Table chunk size. Default: `None`.
- **table_custom_separator**: Custom separator for tables. Default: `None`.
- **table_sniffer**: Whether to automatically detect table separators. Default: `True`.
- **dpi**: Image resolution in DPI. Default: `200`.
- **verbose**: Whether to enable detailed output. Default: `True`.
- **stream**: Whether to use stream processing. Default: `False`.
- **docx_extract_headers_footers**: Whether to extract headers and footers from DOCX files. Default: `True`.
- **docx_extract_images**: Whether to extract images from DOCX files. Default: `True`.
- **pptx_extract_images**: Whether to extract images from PPTX files. Default: `True`.
- **excel_extract_images**: Whether to extract images from Excel files. Default: `True`.
- **excel_max_rows**: Maximum number of rows to extract from Excel files. Default: `None`.
- **max_new_tokens**: Maximum generation length. Default: `8192`.
- **image_batch_size**: Batch size for image processing. Default: `1`.
- **ocr_batch_size**: Batch size for text (OCR) processing. Default: `1`.
