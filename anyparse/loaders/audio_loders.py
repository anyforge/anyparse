import os
import io
import base64
import httpx
import traceback
from typing import Tuple,Union,List,BinaryIO
import librosa
import soundfile
import numpy as np


def autoload_audio(
    file: str 
        | os.PathLike 
        | BinaryIO 
        | bytes 
        | List[str] 
        | List[os.PathLike] 
        | List[BinaryIO] 
        | List[bytes], 
    sample_rate: int | float | None = 22050,
    dtype: np.dtype | str | None = "float32", # ['float32', 'float64', 'int16', 'int32']
    timeout: float | None = None
) -> List[Tuple[np.ndarray, int | float]]:
    if not isinstance(file, list):
        file = [file]  
    speechs = []
    for idx,speech in enumerate(file):
        if isinstance(speech, str):
            if speech.startswith("http://") or speech.startswith("https://"):
                speechdata = io.BytesIO(
                    httpx.get(
                        speech,
                        timeout=timeout, 
                        follow_redirects=True
                    ).content
                )
                speech, sr = librosa.load(
                    speechdata,
                    sr = sample_rate,
                    dtype = dtype
                )
            elif os.path.isfile(speech):
                speech, sr = librosa.load(
                    speech,
                    sr = sample_rate,
                    dtype = dtype
                )
            else:
                # base64字符串
                if speech.startswith("data:audio/"):
                    speech = speech.split(",")[1]
                try:
                    audio_bytes = base64.b64decode(speech)
                    speechdata = io.BytesIO(audio_bytes)
                    speech, sr = librosa.load(
                        speechdata,
                        sr = sample_rate,
                        dtype = dtype
                    )
                except:
                    traceback.print_exc()
                    raise Exception("Incorrect speech source")
                
        elif isinstance(speech, os.PathLike):
            speech, sr = librosa.load(
                speech,
                sr = sample_rate,
                dtype = dtype
            )

        elif isinstance(speech, io.IOBase | io.BufferedIOBase):
            speech, sr = librosa.load(
                speech,
                sr = sample_rate,
                dtype = dtype
            )
            
        elif isinstance(speech, bytes):
            speechdata = io.BytesIO(speech)
            speech, sr = librosa.load(
                speechdata,
                sr = sample_rate,
                dtype = dtype
            )
            
        else:
            raise Exception("Incorrect speech source")
        
        speechs.append(
            (speech, sr)
        )
    return speechs     


def load_audio_to_base64(
    file: str 
        | os.PathLike 
        | BinaryIO 
        | bytes, 
    timeout: float | None = None,
    dtype: np.dtype | str | None = "float32"
) -> dict:   
    """
    加载音频并返回 Base64 编码字符串列表。
    支持本地路径、URL、bytes、文件流等多种输入格式。
    """
    audio_input = file
    res = {}
    try:
        audio_bytes = None
        
        # 1. 处理字符串（URL 或 本地文件路径）
        if isinstance(audio_input, str):
            if audio_input.startswith("http://") or audio_input.startswith("https://"):
                # 从网络获取音频字节流
                response = httpx.get(audio_input, timeout=timeout, follow_redirects=True)
                response.raise_for_status()
                audio_bytes = response.content
            elif os.path.isfile(audio_input):
                # 从本地读取音频字节流
                with open(audio_input, "rb") as f:
                    audio_bytes = f.read()
            else:
                raise ValueError(f"文件不存在或路径无效: {audio_input}")
        
        # 2. 处理 os.PathLike 对象
        elif isinstance(audio_input, os.PathLike):
            with open(audio_input, "rb") as f:
                audio_bytes = f.read()
        
        # 3. 处理文件流对象 (BytesIO, BufferedReader 等)
        elif isinstance(audio_input, (io.IOBase, BinaryIO)):
            # 读取当前指针位置到末尾的所有字节
            audio_bytes = audio_input.read()
        
        # 4. 处理直接的 bytes 字节数据
        elif isinstance(audio_input, bytes):
            audio_bytes = audio_input
        
        else:
            raise TypeError(f"不支持的音频输入类型: {type(audio_input)}")
        
        # 将读取到的二进制音频数据编码为 Base64 字符串
        # base64_str = base64.b64encode(audio_bytes).decode("utf-8")
        with io.BytesIO(audio_bytes) as audio_file:
            with soundfile.SoundFile(audio_file) as f:
                audio_array = f.read(dtype=dtype)
                original_sr = f.samplerate
                audio_format = f.format
                sampling_rate = original_sr
        buffer = io.BytesIO()
        soundfile.write(buffer, audio_array, sampling_rate, format=audio_format.upper())
        buffer.seek(0)
        res = {
            "data": base64.b64encode(buffer.read()).decode("utf-8"),
            "format": audio_format.lower(),            
        }
    except Exception as e:
        traceback.print_exc()
        raise Exception("Incorrect speech source")
    
    return res