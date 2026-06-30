import os
import io
import cv2
import math
import base64
import httpx
import traceback
from typing import Union,List,BinaryIO
import numpy as np
from io import BytesIO
import requests as rq
from PIL import Image, ImageOps


def transpose_by_exif(image):
    """按照图片自带的exif信息矫正"""
    image = ImageOps.exif_transpose(image) or image
    return image


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    

def pillow_to_cv2(image):
    rgb_image = np.array(image)
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    return bgr_image
    
    
def cv2_to_pillow(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    return pil_image


def base64_to_cv2(b64str):
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data


def cv2_to_base64(image):
    # return base64.b64encode(image)
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')


def pillow_to_base64(image):
    img_buffer = BytesIO()
    if image.mode == 'RGBA':
        image.save(img_buffer, format='PNG')
    else:
        image.save(img_buffer, format='JPEG')
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode("utf-8")
    return base64_str


def base64_to_pillow(base64_str):
    image = base64.b64decode(base64_str)
    image = BytesIO(image)
    image = Image.open(image)
    return image


def read_img_from_url(img_url):
    response = rq.get(img_url) #图片地址
    response = response.content
    BytesIOObj = BytesIO()
    BytesIOObj.write(response)
    img = Image.open(BytesIOObj)
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    return img


def autoload_image(
    file: str 
        | os.PathLike 
        | Image.Image 
        | np.ndarray
        | BinaryIO 
        | bytes 
        | List[str] 
        | List[os.PathLike] 
        | List[Image.Image] 
        | List[np.ndarray]
        | List[BinaryIO] 
        | List[bytes], 
    return_rgb: bool = True,
    timeout: float | None = None
) -> Union[Image.Image, List[Image.Image]]:
    if not isinstance(file, list):
        file = [file]  
    images = []
    for idx,image in enumerate(file):
        if isinstance(image, str):
            if image.startswith("http://") or image.startswith("https://"):
                imgdata = io.BytesIO(
                    httpx.get(
                        image,
                        timeout=timeout, 
                        follow_redirects=True
                    ).content
                )
                image = Image.open(imgdata)
            elif os.path.isfile(image):
                image = Image.open(image)
            else:
                # base64字符串
                if image.startswith("data:image/"):
                    image = image.split(",")[1]
                try:
                    imgdata = base64.decodebytes(
                        image.encode(encoding="utf-8")
                    )
                    imgdata = io.BytesIO(imgdata)
                    image = Image.open(imgdata)
                except:
                    traceback.print_exc()
                    raise Exception("Incorrect image source")
        elif isinstance(image, os.PathLike):
            image = Image.open(image)
        elif isinstance(image, Image.Image):
            image = image
        elif isinstance(image, np.ndarray):
            image = image.astype(np.uint8)
            image = Image.fromarray(image)
        elif isinstance(image, io.IOBase | io.BufferedIOBase):
            image = Image.open(image)
        elif isinstance(image, bytes):
            imgdata = io.BytesIO(image)
            image = Image.open(imgdata)
        else:
            raise Exception("Incorrect image source")
        if return_rgb and image.mode != "RGB":
            image = image.convert("RGB")
        images.append(image)
    return images


def resize_image_if_need(image, max_dimension=1124):
    original_width, original_height = image.size
    # 检查是否需要缩放
    width_exceeds = original_width > max_dimension
    height_exceeds = original_height > max_dimension

    if not width_exceeds and not height_exceeds:
        return image
    # 计算缩放比例
    scale_factor = 1.0
    info = ''
    if width_exceeds and height_exceeds:
        # 两个都超过，取较小的缩放比例
        scale_factor = min(max_dimension / original_width, 
                        max_dimension / original_height)
        info = f"宽度和高度都超过{max_dimension}，缩放比例: {scale_factor:.3f}"
    elif width_exceeds:
        # 只有宽度超过
        scale_factor = max_dimension / original_width
        info = f"宽度超过{max_dimension}，缩放比例: {scale_factor:.3f}"
    elif height_exceeds:
        # 只有高度超过
        scale_factor = max_dimension / original_height
        info = f"高度超过{max_dimension}，缩放比例: {scale_factor:.3f}"

    # 计算新尺寸
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # 使用高质量的重采样方法进行缩放
    resized_img = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_img


def smart_resize(
    t: int,
    h: int,
    w: int,
    t_factor: int = 1,
    h_factor: int = 28,
    w_factor: int = 28,
    min_pixels: int = 112 * 112,
    max_pixels: int = 14 * 14 * 4 * 15000,
):
    """
    Smart resize for images.

    Ensures:
    1. Height and width are divisible by the given factors
    2. Total pixels are within [min_pixels, max_pixels]
    3. Keeps aspect ratio as much as possible

    Args:
        t: Temporal dimension.
        h: Height.
        w: Width.
        t_factor: Temporal factor.
        h_factor: Height factor.
        w_factor: Width factor.
        min_pixels: Minimum pixels.
        max_pixels: Maximum pixels.

    Returns:
        (new_h, new_w)
    """
    assert t >= t_factor, "Temporal dimension must be greater than the factor."

    h_bar = round(h / h_factor) * h_factor
    w_bar = round(w / w_factor) * w_factor
    t_bar = round(t / t_factor) * t_factor

    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((t * h * w) / max_pixels)
        h_bar = math.floor(h / beta / h_factor) * h_factor
        w_bar = math.floor(w / beta / w_factor) * w_factor
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (t * h * w))
        h_bar = math.ceil(h * beta / h_factor) * h_factor
        w_bar = math.ceil(w * beta / w_factor) * w_factor

    return h_bar, w_bar


def load_image_to_base64(
    image_source,
    t_patch_size: int,
    max_pixels: int,
    image_format: str,
    patch_expand_factor: int = 1,
    min_pixels: int = 112 * 112,
):
    """Load an image and convert it to base64.

    Supported inputs:
    - PIL.Image.Image
    - Local file path (str)
    - data:image/... URL (str)
    - <|base64|>... blob (str)
    - <|tarpath|>... blob (str)
    - Raw bytes (bytes)

    Args:
        image_source: Image source.
        t_patch_size: Temporal patch size.
        max_pixels: Max pixels.
        image_format: Image format.
        patch_expand_factor: Patch expand factor.
        min_pixels: Min pixels.

    Returns:
        Base64-encoded image content.
    """

    def _try_decode_base64_to_image_bytes(s: str) -> bytes | None:
        # Remove whitespace/newlines and pad for base64.
        candidate = "".join(str(s).split())
        if len(candidate) < 32:
            return None

        # Strip optional "<|base64|>" prefix.
        if candidate.startswith("<|base64|>"):
            candidate = candidate[len("<|base64|>") :]

        # If it looks like a filename (has a short extension), skip.
        if "." in candidate and len(candidate.rsplit(".", 1)[-1]) <= 5:
            return None

        pad = (-len(candidate)) % 4
        if pad:
            candidate = candidate + ("=" * pad)

        try:
            return base64.b64decode(candidate, validate=True)
        except Exception:
            return None

    # Handle different input types
    if isinstance(image_source, Image.Image):
        # Already a PIL Image
        image = image_source
    elif isinstance(image_source, bytes):
        # Raw bytes
        image = Image.open(io.BytesIO(image_source))
    elif isinstance(image_source, str):
        if image_source.startswith("file://"):
            image_source = image_source[7:]

        if os.path.isfile(image_source):
            # Local file path (PDFs are handled via PageLoader)
            with open(image_source, "rb") as f:
                image_data = f.read()
            image = Image.open(io.BytesIO(image_data))
        elif image_source.startswith("data:image/"):
            # data:image/... URL
            image_data = base64.b64decode(image_source.split(",")[1])
            image = Image.open(io.BytesIO(image_data))
        else:
            # Raw base64 payload or <|base64|> blob
            decoded = _try_decode_base64_to_image_bytes(image_source)
            if decoded is None:
                raise ValueError(f"Invalid image source: {image_source}")
            image = Image.open(io.BytesIO(decoded))
    else:
        raise TypeError(f"Unsupported image source type: {type(image_source)}")

    # Convert to RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Original size
    w, h = image.size

    # Compute new size
    h_bar, w_bar = smart_resize(
        t=t_patch_size,
        h=h,
        w=w,
        t_factor=t_patch_size,
        h_factor=14 * 2 * patch_expand_factor,
        w_factor=14 * 2 * patch_expand_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    # Resize
    image = image.resize((w_bar, h_bar), Image.Resampling.BICUBIC)

    # Encode as bytes
    buffered = io.BytesIO()
    image.save(buffered, format=image_format)
    buffered.seek(0)
    image_data = buffered.getvalue()

    # Convert bytes to base64
    base64_encoded_data = base64.b64encode(image_data)
    image_base64 = base64_encoded_data.decode("utf-8")

    return image_base64


def crop_image_region(image, bbox_2d):
    """Crop an image region given a normalized bbox.

    Args:
        image: PIL Image
        bbox_2d: [x1_norm, y1_norm, x2_norm, y2_norm] normalized to 0-1000

    Returns:
        PIL.Image.Image
    """
    image_width, image_height = image.size
    x1_norm, y1_norm, x2_norm, y2_norm = bbox_2d

    # De-normalize to pixel coordinates
    x1 = int(x1_norm * image_width / 1000)
    y1 = int(y1_norm * image_height / 1000)
    x2 = int(x2_norm * image_width / 1000)
    y2 = int(y2_norm * image_height / 1000)

    return image.crop((x1, y1, x2, y2))


def image_tensor_to_base64(image_tensor, image_format):
    """Convert a torch image tensor to base64.

    Args:
        image_tensor: torch.Tensor, shape (C, H, W)
        image_format: Image format.

    Returns:
        Base64-encoded image.
    """

    if image_tensor.shape[0] != 3:
        raise ValueError("Input tensor is not a 3-channel image.")
    image_array = image_tensor.permute(1, 2, 0).numpy()
    image_array = image_array.astype(np.uint8)
    image = Image.fromarray(image_array)
    buffered = io.BytesIO()
    image.save(buffered, format=image_format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str