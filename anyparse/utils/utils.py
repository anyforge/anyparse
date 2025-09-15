import re
import cv2
import time
import uuid
import base64
import hashlib
import aiohttp
import traceback
import numpy as np
from io import BytesIO
import requests as rq
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def create_uuid():
    return str(uuid.uuid1())


def extract_markdown_images(text):
    # 正则表达式：匹配 ![alt_text](url) 格式
    pattern = r'!\[(.*?[^\\])\]\s*\(\s*([^\s\)]+(?:\s+[^\s\)]+)*)\s*\)'
    matches = re.finditer(pattern, text)
    return matches


def resize_image_if_need(image, max_dimension=1920):
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
    print(info)
    return resized_img


def format_filesize(size_bytes):
    """ 将字节大小转换为 KB, MB, GB, TB 并格式化输出 """
    if size_bytes is None:
        return "0.0/MB"
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    unit_index = 0
    while size_bytes >= 1024 and unit_index < len(units) - 1:
        size_bytes /= 1024.0
        unit_index += 1
    return f"{size_bytes:.2f}/{units[unit_index]}"


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


def encrypt_md5(content,encoding='utf-8'):
    md = hashlib.md5(content.encode(encoding=encoding))
    content = md.hexdigest() # 单纯的MD5加密
    return content


def getDayDiff(day1,day2):
    '''
    s = datetime.datetime.now().strftime("%Y-%m-%d")
    e = arrow.get(s).shift(months=-6).format("YYYY-MM-DD")
    num = getDateDiff(s,e)
    all_date_list = getAllDayPerYear(s,num=num)
    print(all_date_list[::-1])
    '''
    time_array1 = time.strptime(day1, "%Y-%m-%d-%H-%M-%S")
    timestamp_day1 = int(time.mktime(time_array1))
    time_array2 = time.strptime(day2, "%Y-%m-%d-%H-%M-%S")
    timestamp_day2 = int(time.mktime(time_array2))
    result = abs(timestamp_day2 - timestamp_day1) // 60 // 60 / 24
    return result


def getSecDiff(day1,day2):
    '''
    s = datetime.datetime.now().strftime("%Y-%m-%d")
    e = arrow.get(s).shift(months=-6).format("YYYY-MM-DD")
    num = getDateDiff(s,e)
    all_date_list = getAllDayPerYear(s,num=num)
    print(all_date_list[::-1])
    '''
    time_array1 = time.strptime(day1, "%Y-%m-%d %H:%M:%S")
    timestamp_day1 = int(time.mktime(time_array1))
    time_array2 = time.strptime(day2, "%Y-%m-%d %H:%M:%S")
    timestamp_day2 = int(time.mktime(time_array2))
    result = abs(timestamp_day2 - timestamp_day1)
    return result


def xywh2xyxy(bbox):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    if isinstance(bbox,list):
        bbox = np.array(bbox)
    bbox[:,2] = bbox[:,0] + bbox[:,2]
    bbox[:,3] = bbox[:,1] + bbox[:,3]
    return bbox


def xyxy2xywh(bbox):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    if isinstance(bbox,list):
        bbox = np.array(bbox)
    bbox[:,2] = bbox[:,2] - bbox[:,0]
    bbox[:,3] = bbox[:,3] - bbox[:,1]
    return bbox


class customvis(object):
    """自定义画框：支持中文，扫描框，矩形框"""
    _COLORS = np.array([
        0.000, 0.447, 0.741, 0.850, 0.325, 0.098, 0.929, 0.694, 0.125, 0.494,
        0.184, 0.556, 0.466, 0.674, 0.188, 0.301, 0.745, 0.933, 0.635, 0.078,
        0.184, 0.300, 0.300, 0.300, 0.600, 0.600, 0.600, 1.000, 0.000, 0.000,
        1.000, 0.500, 0.000, 0.749, 0.749, 0.000, 0.000, 1.000, 0.000, 0.000,
        0.000, 1.000, 0.667, 0.000, 1.000, 0.333, 0.333, 0.000, 0.333, 0.667,
        0.000, 0.333, 1.000, 0.000, 0.667, 0.333, 0.000, 0.667, 0.667, 0.000,
        0.667, 1.000, 0.000, 1.000, 0.333, 0.000, 1.000, 0.667, 0.000, 1.000,
        1.000, 0.000, 0.000, 0.333, 0.500, 0.000, 0.667, 0.500, 0.000, 1.000,
        0.500, 0.333, 0.000, 0.500, 0.333, 0.333, 0.500, 0.333, 0.667, 0.500,
        0.333, 1.000, 0.500, 0.667, 0.000, 0.500, 0.667, 0.333, 0.500, 0.667,
        0.667, 0.500, 0.667, 1.000, 0.500, 1.000, 0.000, 0.500, 1.000, 0.333,
        0.500, 1.000, 0.667, 0.500, 1.000, 1.000, 0.500, 0.000, 0.333, 1.000,
        0.000, 0.667, 1.000, 0.000, 1.000, 1.000, 0.333, 0.000, 1.000, 0.333,
        0.333, 1.000, 0.333, 0.667, 1.000, 0.333, 1.000, 1.000, 0.667, 0.000,
        1.000, 0.667, 0.333, 1.000, 0.667, 0.667, 1.000, 0.667, 1.000, 1.000,
        1.000, 0.000, 1.000, 1.000, 0.333, 1.000, 1.000, 0.667, 1.000, 0.333,
        0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000,
        0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000,
        0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000,
        1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000, 0.000,
        0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000, 1.000,
        0.000, 0.000, 0.000, 0.143, 0.143, 0.143, 0.286, 0.286, 0.286, 0.429,
        0.429, 0.429, 0.571, 0.571, 0.571, 0.714, 0.714, 0.714, 0.857, 0.857,
        0.857, 0.000, 0.447, 0.741, 0.314, 0.717, 0.741, 0.50, 0.5, 0
    ]).astype(np.float32).reshape(-1, 3)
    
    def __init__(self,fontfile = None):
        self.fontfile = fontfile if fontfile else Path(__file__).parent.parent.joinpath('fonts/simkai.ttf').as_posix()
        

    def draw_box(self,img,boxxes,write=False,conf=0.4,scale = 1.3,horiz_align = "left",thickness = 2):
        """ 
        boxxes = [
            [[x0,y0,x1,y1],labelname,labelid,score],
            [[x0,y0,x1,y1],labelname,labelid,score],
            ...,
        ]
        """
        width, height,_ = img.shape
        default_font_size = max(np.sqrt(height * width) // 90, 10 // scale)
        
        for boxes,label,labelid,score in boxxes:     
            x0 = int(boxes[0])
            y0 = int(boxes[1])
            x1 = int(boxes[2])
            y1 = int(boxes[3])        
            labelid = int(labelid)
            score = float(score)
            if score < conf:
                continue            
            ### 定义框，文字颜色，文字背景颜色
            box_color = tuple((customvis._COLORS[labelid] * 255).astype(np.uint8).tolist())
            text_color = tuple((0,0,0) if np.mean(customvis._COLORS[labelid]) > 0.5 else (255,255,255))
            text_bg_color = tuple((customvis._COLORS[labelid] * 255 * 0.7).astype(np.uint8).tolist())
            
            w,h = x1-x0,y1-y0
            height_ratio = h / np.sqrt(height * width)
            font_size = (np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2) * 0.5 * default_font_size) + 13
            font = ImageFont.truetype(self.fontfile, int(font_size), encoding="utf-8")
            
            # 画扫描框
            cv2.rectangle(img, (x0, y0),(x1, y1),box_color, thickness)
            if not write:
                continue
            ### 写文字：支持中文
            cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pilimg = Image.fromarray(cv2img)
            draw = ImageDraw.Draw(pilimg)
            text = "{}:{:.2f}".format(label, score)
            text_width, text_height = draw.textsize(text,font = font)
            ### 画文字背景
            draw.rectangle([(x0+thickness, y0+thickness),(x0 + text_width+thickness, y0 + text_height+thickness)], fill=text_bg_color)
            ### 写文字
            draw.text((x0,y0), text, text_color, font = font, align = horiz_align)
            img = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
        return img
    
            
    def draw_scan_box(self,img,boxxes,write=False,conf=0.4,scale = 1.3,edge=3,horiz_align = "left",thickness = 2,lineType = 4):
        """ 
        boxxes = [
            [[x0,y0,x1,y1],labelname,labelid,score],
            [[x0,y0,x1,y1],labelname,labelid,score],
            ...,
        ]
        """
        width, height,_ = img.shape
        default_font_size = max(np.sqrt(height * width) // 90, 10 // scale)
        
        for boxes,label,labelid,score in boxxes:  
            x0 = int(boxes[0])
            y0 = int(boxes[1])
            x1 = int(boxes[2])
            y1 = int(boxes[3])        
            labelid = int(labelid)
            score = float(score)
            if score < conf:
                continue            
            ### 定义框，文字颜色，文字背景颜色
            box_color = tuple((customvis._COLORS[labelid] * 255).astype(np.uint8).tolist())
            text_color = tuple((0,0,0) if np.mean(customvis._COLORS[labelid]) > 0.5 else (255,255,255))
            text_bg_color = tuple((customvis._COLORS[labelid] * 255 * 0.7).astype(np.uint8).tolist())
            
            ### 定义扫描框缺省
            min_edge = min(x1-x0,y1-y0)
            min_edge = int(min_edge / edge)
            width, height,_ = img.shape
            w,h = x1-x0,y1-y0
            default_font_size = max(np.sqrt(height * width) // 90, 10 // scale)
            height_ratio = h / np.sqrt(height * width)
            font_size = (np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2) * 0.5 * default_font_size) + 13
            font = ImageFont.truetype(self.fontfile, int(font_size), encoding="utf-8")
            
            # 画扫描框
            ### 左上两边
            img = cv2.line(img,(x0,y0),(x0+min_edge,y0),box_color,thickness, lineType)
            img = cv2.line(img,(x0,y0),(x0,y0+min_edge),box_color,thickness, lineType)
            ### 右下两边
            img = cv2.line(img,(x1,y1),(x1-min_edge,y1),box_color,thickness, lineType)
            img = cv2.line(img,(x1,y1),(x1,y1-min_edge),box_color,thickness, lineType)
            ### 右上两边
            img = cv2.line(img,(x1,y0),(x1-min_edge,y0),box_color,thickness, lineType)
            img = cv2.line(img,(x1,y0),(x1,y0+min_edge),box_color, thickness, lineType)
            ### 左下两边
            img = cv2.line(img,(x0,y1),(x0,y1-min_edge),box_color, thickness, lineType)
            img = cv2.line(img,(x0,y1),(x0+min_edge,y1),box_color, thickness, lineType)
            
            if not write:
                continue
            ### 写文字：支持中文
            cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pilimg = Image.fromarray(cv2img)
            draw = ImageDraw.Draw(pilimg)
            text = "{}:{:.2f}".format(label, score)
            text_width, text_height = draw.textsize(text,font = font)
            ### 画文字背景
            draw.rectangle([(x0+thickness, y0+thickness),(x0 + text_width+thickness, y0 + text_height+thickness)], fill=text_bg_color)
            ### 写文字
            draw.text((x0,y0), text, text_color, font = font, align = horiz_align)
            img = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
        return img
    
    
    def draw_box_pillow(self,img,boxxes,write=False,conf=0.4,scale = 1.3,horiz_align = "left",thickness = 2):
        """ 
        boxxes = [
            [[x0,y0,x1,y1],labelname,labelid,score],
            [[x0,y0,x1,y1],labelname,labelid,score],
            ...,
        ]
        """
        width, height,_ = img.shape
        default_font_size = max(np.sqrt(height * width) // 90, 10 // scale)
        cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pilimg = Image.fromarray(cv2img)
        draw = ImageDraw.Draw(pilimg)
        
        for boxes,label,labelid,score in boxxes:      
            x0 = int(boxes[0])
            y0 = int(boxes[1])
            x1 = int(boxes[2])
            y1 = int(boxes[3])        
            labelid = int(labelid)
            score = float(score)
            if score < conf:
                continue            
            ### 定义框，文字颜色，文字背景颜色
            box_color = tuple((customvis._COLORS[labelid] * 255).astype(np.uint8).tolist())
            text_color = tuple((0,0,0) if np.mean(customvis._COLORS[labelid]) > 0.5 else (255,255,255))
            text_bg_color = tuple((customvis._COLORS[labelid] * 255 * 0.7).astype(np.uint8).tolist())

            w,h = x1-x0,y1-y0
            height_ratio = h / np.sqrt(height * width)
            font_size = (np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2) * 0.5 * default_font_size) + 13
            font = ImageFont.truetype(self.fontfile, int(font_size), encoding="utf-8")
            text = "{}:{:.2f}".format(label, score)
            text_width, text_height = draw.textsize(text,font = font)
            
            # 画扫描框
            draw.rectangle([(x0, y0),(x1, y1)],outline=box_color, width=thickness)
            
            if not write:
                continue
            ### 写文字：支持中文
            ### 画文字背景
            draw.rectangle([(x0+thickness, y0+thickness),(x0 + text_width+thickness, y0 + text_height+thickness)], fill=text_bg_color)
            ### 写文字
            draw.text((x0,y0), text, text_color, font = font, align = horiz_align)
        img = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
        return img
    
            
    def draw_scan_box_pillow(self,img,boxxes,write=False,conf=0.4,scale = 1.3,edge=3,horiz_align = "left",thickness = 2):
        """ 
        boxxes = [
            [[x0,y0,x1,y1],labelname,labelid,score],
            [[x0,y0,x1,y1],labelname,labelid,score],
            ...,
        ]
        """
        width, height,_ = img.shape
        default_font_size = max(np.sqrt(height * width) // 90, 10 // scale)
        cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pilimg = Image.fromarray(cv2img)
        draw = ImageDraw.Draw(pilimg)
        
        for boxes,label,labelid,score in boxxes:        
            x0 = int(boxes[0])
            y0 = int(boxes[1])
            x1 = int(boxes[2])
            y1 = int(boxes[3])        
            labelid = int(labelid)
            score = float(score)
            if score < conf:
                continue
            ### 定义框，文字颜色，文字背景颜色
            box_color = tuple((customvis._COLORS[labelid] * 255).astype(np.uint8).tolist())
            text_color = tuple((0,0,0) if np.mean(customvis._COLORS[labelid]) > 0.5 else (255,255,255))
            text_bg_color = tuple((customvis._COLORS[labelid] * 255 * 0.7).astype(np.uint8).tolist())
            
            ### 定义扫描框缺省
            min_edge = min(x1-x0,y1-y0)
            min_edge = int(min_edge / edge)
        
            w,h = x1-x0,y1-y0
            height_ratio = h / np.sqrt(height * width)
            font_size = (np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2) * 0.5 * default_font_size) + 13
            font = ImageFont.truetype(self.fontfile, int(font_size), encoding="utf-8")
            text = "{}:{:.2f}".format(label, score)
            text_width, text_height = draw.textsize(text,font = font)
            
            # 画扫描框            
            ### 左上两边
            draw.line([(x0,y0),(x0+min_edge,y0)],fill = box_color,width = thickness)
            draw.line([(x0,y0),(x0,y0+min_edge)],fill = box_color,width = thickness)
            ### 右下两边
            draw.line([(x1,y1),(x1-min_edge,y1)],fill = box_color,width = thickness)
            draw.line([(x1,y1),(x1,y1-min_edge)],fill = box_color,width = thickness)
            ### 右上两边
            draw.line([(x1,y0),(x1-min_edge,y0)],fill = box_color,width = thickness)
            draw.line([(x1,y0),(x1,y0+min_edge)],fill = box_color,width = thickness)
            ### 左下两边
            draw.line([(x0,y1),(x0,y1-min_edge)],fill = box_color,width = thickness)
            draw.line([(x0,y1),(x0+min_edge,y1)],fill = box_color,width = thickness)
            
            if not write:
                continue
            ### 写文字：支持中文
            ### 画文字背景
            draw.rectangle([(x0+thickness, y0+thickness),(x0 + text_width+thickness, y0 + text_height+thickness)], fill=text_bg_color)
            ### 写文字
            draw.text((x0,y0), text, text_color, font = font, align = horiz_align)
        img = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
        return img


class Readf():
    def __init__(self,file,encoding = 'utf-8',strip = True):
        self.file=file
        self.strip = strip
        self.encoding=encoding
        
    def __iter__(self):
        try:
            with open(self.file,'r',encoding=self.encoding) as f:
                for i,j in enumerate(f):
                    if self.strip:
                        yield j.strip()
                    else:
                        yield j
        except Exception:
            traceback.print_exc()


class autodlhash(object):
    """哈希加密：'md5', 'sha1', 'sha256'"""
    @classmethod
    def encrypt_string(cls, data, algorithm = 'md5'):
        hash = hashlib.new(algorithm)
        hash.update(data.encode('utf8'))
        return hash.hexdigest()        
        
    @classmethod
    def encrypt_file(cls, fpath, algorithm = 'md5', size = 4096):
        with open(fpath, 'rb') as f:
            hash = hashlib.new(algorithm)
            for chunk in iter(lambda: f.read(size), b''):
                hash.update(chunk)
        return hash.hexdigest()   
    
    
def split_batch(data, batch_size):
    if isinstance(data,np.ndarray):
        for batch in np.array_split(data, batch_size):
            yield batch
    else:
        for idx in range(0, len(data), batch_size): 
            yield data[idx: idx + batch_size]
            

def calculate_iou(bbox1, bbox2):
    """计算两个边界框的交并比(IOU)。

    Args:
        bbox1 (list[float]): 第一个边界框的坐标，格式为 [x1, y1, x2, y2]，其中 (x1, y1) 为左上角坐标，(x2, y2) 为右下角坐标。
        bbox2 (list[float]): 第二个边界框的坐标，格式与 `bbox1` 相同。

    Returns:
        float: 两个边界框的交并比(IOU)，取值范围为 [0, 1]。
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The area of overlap area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # The area of both rectangles
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    if any([bbox1_area == 0, bbox2_area == 0]):
        return 0

    # Compute the intersection over union by taking the intersection area
    # and dividing it by the sum of both areas minus the intersection area
    iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)

    return iou


def latex_rm_whitespace(s: str):
    raw_s = s
    try:
        """Remove unnecessary whitespace from LaTeX code."""
        text_reg = r"(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})"
        letter = "[a-zA-Z]"
        noletter = "[\W_^\d]"
        names = [x[0].replace(" ", "") for x in re.findall(text_reg, s)]
        s = re.sub(text_reg, lambda match: str(names.pop(0)), s)
        news = s
        while True:
            s = news
            news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, noletter), r"\1\2", s)
            news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, letter), r"\1\2", news)
            news = re.sub(r"(%s)\s+?(%s)" % (letter, noletter), r"\1\2", news)
            if news == s:
                break
        return s
    except:
        return raw_s


def sorted_layout_boxes(res, w, is_column = False):
    """
    w: page width
    res: [{"bbox": [x0,y0,x1,y1],'content': []},...]
    """
    num_boxes = len(res)
    if num_boxes == 1:
        return res

    sorted_boxes = sorted(res, key=lambda x: (x['bbox'][1], x['bbox'][0]))
    _boxes = list(sorted_boxes)
    # print("sorted boxes::: ", sorted_boxes)
    if not is_column:
        return _boxes
    new_res = []
    res_left = []
    res_right = []
    i = 0

    while True:
        if i >= num_boxes:
            break
        if i == num_boxes - 1:
            if _boxes[i]['bbox'][1] > _boxes[i - 1]['bbox'][3] and _boxes[i]['bbox'][0] < w / 2 and _boxes[i]['bbox'][2] > w / 2:
                new_res += res_left
                new_res += res_right
                # _boxes[i]['layout'] = 'single'
                new_res.append(_boxes[i])
            else:
                if _boxes[i]['bbox'][2] > w / 2:
                    # _boxes[i]['layout'] = 'double'
                    res_right.append(_boxes[i])
                    new_res += res_left
                    new_res += res_right
                elif _boxes[i]['bbox'][0] < w / 2:
                    # _boxes[i]['layout'] = 'double'
                    # 定义匹配页码的正则表达式
                    text = _boxes[i]['text']
                    page_pattern = re.compile(r'''
                        ^                        # 起始位置
                        (
                            (?:第\s*[IVXLCDM]+\s*页)    |  # 中文罗马数字：第IV页
                            (?:第\s*\d+\s*页)          |  # 中文数字：第1页
                            (?:第\s*[a-zA-Z]\s*页)     |  # 中文字母：第a页
                            (?:page[- ]*[IVXLCDM]+)    |  # 英文罗马数字：page-IV
                            (?:page[- ]*\d+)           |  # 英文数字：page-1/page 1
                            (?:page[- ]*[a-zA-Z])      |  # 英文字母：page-a
                            \d+                       |  # 纯数字：100
                            [IVXLCDM]+                   # 纯罗马数字：IV
                        )
                        $                        # 结束位置
                    ''', re.VERBOSE | re.IGNORECASE)
                    if page_pattern.fullmatch(text):
                        res_right.append(_boxes[i])
                    else:
                        res_left.append(_boxes[i])
                    new_res += res_left
                    new_res += res_right
            res_left = []
            res_right = []
            break
        elif _boxes[i]['bbox'][0] < w / 4 and _boxes[i]['bbox'][2] < 3 * w / 4:
            # _boxes[i]['layout'] = 'double'
            res_left.append(_boxes[i])
            i += 1
        elif _boxes[i]['bbox'][0] > w / 4 and _boxes[i]['bbox'][2] > w / 2:
            # _boxes[i]['layout'] = 'double'
            if _boxes[i]['bbox'][0] < ((w/2) * 0.95):
                res_left.append(_boxes[i])
            else:
                res_right.append(_boxes[i])
            i += 1
        else:
            new_res += res_left
            new_res += res_right
            # _boxes[i]['layout'] = 'single'
            new_res.append(_boxes[i])
            res_left = []
            res_right = []
            i += 1
    if res_left:
        new_res += res_left
    if res_right:
        new_res += res_right
    return new_res   