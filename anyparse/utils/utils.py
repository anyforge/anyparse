import re
import uuid
import hashlib
import traceback
import numpy as np


def create_uuid():
    return str(uuid.uuid1())


def clean_text_linebreak(text):
    text = re.sub(r'\n{2,}', '\n', text).strip()
    return text


def extract_markdown_images(text):
    # 正则表达式：匹配 ![alt_text](url) 格式
    pattern = r'!\[(.*?[^\\])\]\s*\(\s*([^\s\)]+(?:\s+[^\s\)]+)*)\s*\)'
    matches = re.finditer(pattern, text)
    return matches


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


def encrypt_md5(content,encoding='utf-8'):
    md = hashlib.md5(content.encode(encoding=encoding))
    content = md.hexdigest() # 单纯的MD5加密
    return content


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

