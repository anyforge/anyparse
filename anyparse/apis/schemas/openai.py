import json
import datetime
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_serializer
from typing import List, Optional, Dict, Any, Union, Literal
from fastapi.responses import StreamingResponse
from ..views.contexts import default_maxsize


class BaseDataModel(BaseModel):
    model_config = ConfigDict(
        extra='allow',
        use_enum_values=True,
        arbitrary_types_allowed=True,
        # V2 中 json_encoders 的用法保持不变
        json_encoders={
            datetime.datetime: lambda v: v.isoformat(),
            datetime.date: lambda v: v.isoformat(),
            datetime.time: lambda v: v.isoformat(),
        }
    )
    

# OpenAI Chat API
class ModelObject(BaseDataModel):
    id: str
    object: str = "model"
    created: int = int(datetime.datetime.now().timestamp())
    owned_by: str = "model"
    description: str = "Any file parse to markdown"
    allow_mimetypes: Dict = {}


class ModelListResponse(BaseDataModel):
    object: str = "list"
    data: List[ModelObject]


class ChatMessageFileContentPart(BaseDataModel):
    file_name: str = ''
    file_data: Optional[str] = None


class ChatMessageContentPart(BaseDataModel):
    """消息内容片段，支持 text 和 file 类型"""
    type: str = Field(..., description="内容类型: 'text' 或 'file' 或者 'file_id'")
    text: Optional[str] = None
    file: Optional[ChatMessageFileContentPart] = ChatMessageFileContentPart()
    
    @model_serializer(mode='wrap')
    def serialize_with_truncated_file(self, handler):
        # 1. 先调用默认的序列化逻辑，生成基础字典
        data = handler(self)
        
        # 2. 检查并截断 file_data
        if data.get("file") and isinstance(data["file"].get("file_data"), str):
            data["file"]["file_data"] = data["file"]["file_data"][:100]
            
        return data


class ChatMessage(BaseDataModel):
    role: Literal["user", "assistant", "system", "tool"]
    content: Union[str, dict, List[ChatMessageContentPart]] = Field(..., description="消息内容")
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    

class ChatUsageInfo(BaseDataModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionChoice(BaseDataModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter", "null"]] = None


class ChatCompletionResponse(BaseDataModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatUsageInfo


class ChatCompletionChunkDelta(BaseDataModel):
    role: Optional[Literal["assistant"]] = None
    content: Optional[str | list | dict] = None
    tool_calls: Optional[List[Any]] = None  # 简化处理，如需可细化


class ChatCompletionChunkChoice(BaseDataModel):
    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter", "null"]] = None


class ChatCompletionChunkResponse(BaseDataModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]


class ChatCompletionRequest(BaseDataModel):
    model: str = Field(default="agent", description="模型名称")
    messages: List[ChatMessage] = Field(..., min_items=1)
    stream: bool = False
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, gt=0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    stop: Optional[Union[str, List[str]]] = None
    # 允许任意额外字段（如 interrupt_before, thread_id, tools 等）
    runtimes_args: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("messages")
    def validate_messages(cls, v):
        if not v:
            raise ValueError("messages cannot be empty")
        return v
    

def chat_format_openai_finish_chunk(request_id: str, model: str) -> str:
    """流式返回结果构建结束标志"""
    
    response = ChatCompletionChunkResponse(
        id=request_id,
        created=int(datetime.datetime.now().timestamp()),
        model=model,
        choices=[{
            "index": 0,
            "delta": {},  # ✅ 必须是空对象，不能有任何字段
            "finish_reason": "stop"
        }]
    )
    json_str = json.dumps(
        response.model_dump(exclude_none=True),
        ensure_ascii=False,
    )
    return f"data: {json_str}\n\n"


def chat_format_openai_stream_chunk(chunk: ChatCompletionChunkDelta, request_id: str, model: str) -> str:
    """流式返回结果chunk"""
    delta = ChatCompletionChunkDelta(role="assistant", content=chunk.content or {})
    choice = ChatCompletionChunkChoice(index=0, delta=delta, finish_reason=None)
    response = ChatCompletionChunkResponse(
        id=request_id,
        created=int(datetime.datetime.now().timestamp()),
        model=model,
        choices=[choice]
    )
    json_str = json.dumps(
        response.model_dump(exclude_none=True),
        ensure_ascii=False,
    )
    return f"data: {json_str}\n\n"


def chat_format_openai_final_response(
    total_content: str | dict, 
    request_id: str, 
    model: str,
    total_tokens: int
) -> ChatCompletionResponse:
    """非流式返回结果"""
    message = ChatMessage(role="assistant", content=total_content or {})
    choice = ChatCompletionChoice(index=0, message=message, finish_reason="stop")
    usage = ChatUsageInfo(
        prompt_tokens = 0,
        completion_tokens = total_tokens,
        total_tokens = total_tokens
    )
    return ChatCompletionResponse(
        id=request_id,
        created=int(datetime.datetime.now().timestamp()),
        model=model,
        choices=[choice],
        usage=usage
    )
    

def chat_format_openai_response_chunk(
    total_content: str | dict, 
    request_id: str, 
    model: str,    
):
    """流式返回失败的结果chunk"""
    chunk = ChatCompletionChunkDelta(
        role="assistant",
        content=total_content
    )
    content = chat_format_openai_stream_chunk(
        chunk = chunk, 
        request_id = request_id, 
        model = model
    )
    yield content                                          
    # 发送结束标记
    content = chat_format_openai_finish_chunk(
            request_id = request_id, 
            model = model
    )
    yield content
 
    
def chat_format_openai_response_sse(
    total_content: str | dict, 
    request_id: str, 
    model: str,   
):
    """流式返回失败的结果"""
    output = StreamingResponse(
        chat_format_openai_response_chunk(
            total_content = total_content,
            request_id = request_id, 
            model = model
        ), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )              
    return output   


# OpenAI Responses API
class ResponsesMessageFileContentPart(BaseDataModel):
    file_name: str = ''
    file_data: Optional[str] = None


class ResponsesMessageContentPart(BaseDataModel):
    """消息内容片段，支持 text 和 file 类型"""
    type: str = Field(..., description="内容类型: 'text' 或 'file' 或者 'file_id'")
    text: Optional[str] = None
    file: Optional[ResponsesMessageFileContentPart] = ResponsesMessageFileContentPart()
    
    @model_serializer(mode='wrap')
    def serialize_with_truncated_file(self, handler):
        # 1. 先调用默认的序列化逻辑，生成基础字典
        data = handler(self)
        
        # 2. 检查并截断 file_data
        if data.get("file") and isinstance(data["file"].get("file_data"), str):
            data["file"]["file_data"] = data["file"]["file_data"][:100]
            
        return data


class ResponsesInputMessage(BaseDataModel):
    """Responses API 的输入消息"""
    role: Literal["user", "assistant", "system"] = "user"
    content: Union[str, dict, List[ResponsesMessageContentPart]] = Field(..., description="输入内容")


class ResponsesAPIRequest(BaseDataModel):
    """Responses API 请求体"""
    model: str = Field(default="agent", description="模型名称")
    input: Union[str, List[ResponsesInputMessage]] = Field(..., description="对话输入")
    instructions: Optional[str] = Field(None, description="系统级指令")
    stream: bool = False
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_output_tokens: Optional[int] = Field(None, gt=0)
    # 允许任意额外字段
    runtimes_args: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("input")
    def validate_input(cls, v):
        if not v:
            raise ValueError("input cannot be empty")
        return v


class ResponsesOutputTextContent(BaseDataModel):
    """输出内容片段"""
    type: Literal["output_text"] = "output_text"
    text: Any  # ✅ 完美兼容你的 dict 业务数据


class ResponsesOutputMessage(BaseDataModel):
    """Responses API 的输出消息"""
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: List[ResponsesOutputTextContent] = Field(..., description="输出内容列表")
    status: Literal["completed", "in_progress"] = "completed"


class ResponsesAPIResponse(BaseDataModel):
    """Responses API 非流式响应体"""
    id: str
    object: Literal["response"] = "response"
    created_at: int = Field(default_factory=lambda: int(datetime.datetime.now().timestamp()))
    model: str
    output: List[ResponsesOutputMessage]
    status: Literal["completed", "in_progress", "failed"] = "completed"
    usage: Optional[Dict[str, int] | ChatUsageInfo] = None


class ResponsesStreamEvent(BaseDataModel):
    """流式事件基类"""
    type: str
    response_id: str
    model: str  # ✅ 将 model 字段加入基类，所有事件都会继承


class ResponsesTextDeltaEvent(ResponsesStreamEvent):
    """文本增量事件"""
    type: Literal["response.output_text.delta"] = "response.output_text.delta"
    delta: Any  # ✅ 允许 str 或 dict


class ResponsesTextDoneEvent(ResponsesStreamEvent):
    """文本结束事件"""
    type: Literal["response.completed"] = "response.completed"


def responses_format_openai_finish_chunk(request_id: str, model: str) -> str:
    """流式返回结果构建结束标志"""
    # Responses API 的结束事件
    event = ResponsesTextDoneEvent(
        response_id=request_id, 
        model=model
    )
    json_str = json.dumps(
        event.model_dump(exclude_none=True),
        ensure_ascii=False,
    )
    return f"event: response.completed\ndata: {json_str}\n\n"


def responses_format_openai_stream_chunk(chunk_content: str | dict, request_id: str, model: str) -> str:
    """流式返回结果chunk"""
    # 将内容包装为 delta 事件
    event = ResponsesTextDeltaEvent(
        response_id=request_id, 
        model=model, 
        delta=json.dumps(chunk_content,ensure_ascii=False)
    )
    json_str = json.dumps(
        event.model_dump(exclude_none=True),
        ensure_ascii=False,
    )
    return f"event: response.output_text.delta\ndata: {json_str}\n\n"


def responses_format_openai_final_response(
    total_content: str | dict, 
    request_id: str, 
    model: str,
    total_tokens: int = 0,
    status: Literal["completed", "failed"] = "completed"
) -> ResponsesAPIResponse:
    """非流式返回结果"""
    # 完美兼容你的 dict 业务数据
    # content_item = {"type": "output_text", "text": total_content}
    content_item = ResponsesOutputTextContent(
        text=json.dumps(total_content,ensure_ascii=False)
        # text=total_content
    )
    output_message = ResponsesOutputMessage(
        content=[content_item],
        status="completed" if status == "completed" else "in_progress"
    )
    
    usage = ChatUsageInfo(
        prompt_tokens = 0,
        completion_tokens = total_tokens,
        total_tokens = total_tokens
    )
    return ResponsesAPIResponse(
        id=request_id,
        model=model,
        output=[output_message],
        status=status,
        usage=usage
    )


def responses_format_openai_response_chunk(
    total_content: str | dict, 
    request_id: str, 
    model: str,    
):
    """流式返回失败/成功的结果chunk"""
    # 1. 发送内容 chunk
    content = responses_format_openai_stream_chunk(
        chunk_content=total_content, 
        request_id=request_id, 
        model=model
    )
    yield content                                          
    
    # 2. 发送结束标记
    finish_content = responses_format_openai_finish_chunk(
        request_id=request_id, 
        model=model
    )
    yield finish_content


def responses_format_openai_response_sse(
    total_content: str | dict, 
    request_id: str, 
    model: str,   
) -> StreamingResponse:
    """流式返回结果 (SSE)"""
    return StreamingResponse(
        responses_format_openai_response_chunk(
            total_content=total_content,
            request_id=request_id, 
            model=model
        ), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# OpenAI Files API
class FileObject(BaseDataModel):
    id: str
    object: str = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: str
    deleted: Optional[bool] = None 
    code: int
    msg: str
    
    
# OpenAI extra body
class RuntimesArgs(BaseDataModel):
    file_idx: str = ''
    maxsize: float = default_maxsize
    autodetect_encoding: bool = True
    use_doc_cls: bool = False
    use_doc_rectifier: bool = False
    use_doc_layout: bool = True
    doc_layout_image_min_size: int = 500
    text_encoding: str = "utf-8"
    table_chunk_size: int = 4096
    table_custom_separator: str = None
    dpi: int = 200
    verbose: bool = True
    stream: bool = False
    autocal_md5: bool = False
    docx_extract_headers_footers: bool = True
    docx_extract_images: bool = True
    pptx_extract_images: bool = True
    excel_extract_images: bool = True
    excel_max_rows: int = None
    use_image_resize: bool = True
    max_new_tokens: int = 16384
    image_batch_size: int = 1
    ocr_batch_size: int = 1  
