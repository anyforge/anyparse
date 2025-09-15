import io
import time
import datetime
import traceback
import pymupdf
from PIL import Image
from shapely.geometry import Polygon
from ..constructs.pdfclassify import pdfClassify
from ..utils.utils import sorted_layout_boxes


class pdfConverter(object):
    def __init__(self, config = None):
        self.config = config
        self.pdfclf = pdfClassify()

    def process_tags(self, content: str) -> str:
        """Replaces special tags with HTML entities to prevent them from being rendered as HTML."""
        content = content.replace("<img>", "&lt;img&gt;")
        content = content.replace("</img>", "&lt;/img&gt;")
        content = content.replace("<watermark>", "&lt;watermark&gt;")
        content = content.replace("</watermark>", "&lt;/watermark&gt;")
        content = content.replace("<page_number>", "&lt;page_number&gt;")
        content = content.replace("</page_number>", "&lt;/page_number&gt;")
        content = content.replace("<signature>", "&lt;signature&gt;")
        content = content.replace("</signature>", "&lt;/signature&gt;")
        return content 
    
    def invoke_image(self, file, ocrmodel, ocr_mode = "base", **kwargs):
        res = []
        images = {}
        full_text = ""
        rawimage = Image.open(file)
        try:
            tt = time.time()
            image = rawimage.copy()
            image_content = ocrmodel.invoke(image, mode = ocr_mode, **kwargs)
            image_content = image_content.to_markdown()        
            time_res = time.time() - tt
  
        except:       
            traceback.print_exc() 
            tt = time.time()  
            image = rawimage.copy()     
            image_content = ocrmodel.invoke(image, mode = "base", **kwargs)
            image_content = image_content.to_markdown()
            time_res = time.time() - tt
        finally:
            if full_text.endswith('\n'):
                full_text += f"{image_content}\n"
            else:
                full_text += f"\n{image_content}\n"    
            res.append({
                "type": "image",
                "content": full_text,
                "images": images,
                "time_elapse": time_res
            })
            return res      
        
    async def ainvoke_image_stream(self, file, ocrmodel, **kwargs):
        rawimage = Image.open(file)
        image = rawimage.copy()
        outputs = await ocrmodel.ainvoke_stream(image, **kwargs)
        return outputs
    
    async def ainvoke_pdf_stream(self, file, ocrmodel, parsecallback = None, **kwargs):
        doc = pymupdf.open(file.as_posix())
        total_page_count = doc.page_count
        # 处理每一页
        time_elapse = []
        if parsecallback:
            parseinfo = {
                "file": file.name,
                "total_pages": total_page_count,
                "start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            parsecallback.on_started(**parseinfo)
        for page_number in range(total_page_count):
            yield f"[PAGEDONE]:{page_number + 1}"
            tt = time.time()
            page = doc.load_page(page_number)
            page_width = page.rect.width        
            pix = page.get_pixmap(dpi = 72)
            img_bytes = pix.tobytes("png")
            # 使用BytesIO读取字节流，并使用Pillow打开图像
            pil_image = Image.open(io.BytesIO(img_bytes))
            outputs = await ocrmodel.ainvoke_stream(pil_image, **kwargs)
            async for item in outputs:
                yield item
            time_res = time.time() - tt
            time_elapse.append(time_res)
            if parsecallback:
                parseinfo = {
                    "file": file.name,
                    "total_pages": total_page_count,
                    "finish_page": page_number + 1,
                    "page_elapse_time": time_res,
                    "finish_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                parsecallback.on_page_parsed(**parseinfo)
                
        if parsecallback:
            parseinfo = {
                "file": file.name,
                "total_pages": total_page_count,
                "total_elapse_time": sum(time_elapse),
                "finish_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            parsecallback.on_finished(**parseinfo)
        return
    
    def invoke_pdf(self, file, ocrmodel, pdf_mode = "auto", ocr_mode = "base", parsecallback = None, **kwargs):
        is_column = kwargs.get("is_column", False)
        res = []
        images = {}
        doc = pymupdf.open(file.as_posix())
        total_page_count = doc.page_count
        # 处理每一页
        time_elapse = []
        if parsecallback:
            parseinfo = {
                "file": file.name,
                "total_pages": total_page_count,
                "start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            parsecallback.on_started(**parseinfo)
        pdf_classify_result = {}
        for page_number in range(total_page_count):
            full_text = ""
            page = doc.load_page(page_number)
            page_width = page.rect.width
            text_blocks = page.get_text('blocks')
            tt = time.time()
            try:
                ### 判定pdf页面类型：文本或者ocr
                pdf_classify_result = self.pdfclf.invoke(page, pdf_mode=pdf_mode, **kwargs)
                ### 文图型pdf进行正常解析
                if pdf_classify_result.label == "txt":
                    image_blocks = page.get_images(full=True)
                    table_blocks = page.find_tables().tables
                    
                    text_list = []
                    for idx,block in enumerate(text_blocks):
                        try:
                            tx0,ty0,tx1,ty1 = block[:4]
                            text = block[4]
                            text_list.append((tx0,ty0,tx1,ty1,text,'text',idx))
                        except:
                            traceback.print_exc()
                    
                    table_list = []
                    for idx,block in enumerate(table_blocks):
                        try:
                            box = block.bbox
                            x1,y1,x2,y2 = box
                            table_content = ['|'.join([str(x) for x in row if x]) for row in block.extract()]
                            table_content = '\n'.join(table_content)
                            table_list.append((x1,y1,x2,y2,table_content,'table',idx))
                        except:
                            traceback.print_exc()
                    
                    image_list = []
                    for idd,img in enumerate(image_blocks):
                        try:
                            xref = img[0]
                            name = img[7]
                            # img_rect = page.get_image_bbox(xref)
                            rect = page.get_image_rects(xref)
                            image_data = doc.extract_image(xref).get('image','')
                            if len(rect) > 0 and image_data:
                                img_rect = rect[0]
                                ix0, iy0 = img_rect.tl  # 左上角
                                ix1, iy1 = img_rect.br  # 右下角                     
                
                                image = Image.open(io.BytesIO(image_data))
                                image_info = (ix0,iy0,ix1,iy1,image,'image',f"page{page_number}_xref{xref}_name{name}.png")
                                if image_info not in image_list:
                                    image_list.append(image_info)
                        except:
                            traceback.print_exc()
                    new_text_list = []
                    for idx,line in enumerate(text_list):
                        flag = False
                        tx0,ty0,tx1,ty1,text,label1,idd = line
                        text_poly = [(tx0,ty0), (tx1,ty0), (tx1,ty1), (tx0,ty1)]
                        text_poly = Polygon(text_poly)
                        for tbline in table_list:
                            x1,y1,x2,y2,table_content,label2,ide = tbline
                            table_poly = [(x1,y1), (x2,y1), (x2,y2), (x1,y2)]
                            table_poly = Polygon(table_poly)
                            if text_poly.intersects(table_poly):
                                flag = True
                                break
                        if not flag:
                            new_text_list.append((tx0,ty0,tx1,ty1,text,'text',idx))
                    # page_list = list(sorted(new_text_list+image_list+table_list,key = lambda x: (x[1],x[0])))
                    page_list = []
                    need_sorted_page_list = new_text_list+image_list+table_list
                    for line in need_sorted_page_list:
                        bbox = line[:4]
                        text = line[4]
                        content = line
                        page_list.append({
                            "bbox": bbox,
                            "text": text,
                            "content": content,
                        })
                    page_list = sorted_layout_boxes(page_list,page_width, is_column=is_column)
                    for line in page_list:
                        line = line['content']
                        content = line[-3]
                        content_type = line[-2]
                        content_name = line[-1]
                        if content_type in ['text','table']:
                            if full_text.endswith('\n'):
                                full_text += f'{content}\n'
                            else:
                                full_text += f'\n{content}\n'
                        elif content_type == 'image':
                            pil_image = content
                            image_content = ocrmodel.invoke(pil_image, mode = ocr_mode, **kwargs)
                            image_content = image_content.to_markdown()
                            images[content_name] = {
                                "image": content,
                                "content": image_content
                            }
                            if image_content:
                                if full_text.endswith('\n'):
                                    full_text += f"{image_content}\n"
                                else:
                                    full_text += f"\n{image_content}\n"
                              
                ### 纯图型pdf进行miner解析        
                else:
                    pix = page.get_pixmap(dpi = 72)
                    img_bytes = pix.tobytes("png")
                    # 使用BytesIO读取字节流，并使用Pillow打开图像
                    pil_image = Image.open(io.BytesIO(img_bytes))
                    print(pil_image.size)
                    image_content = ocrmodel.invoke(pil_image, mode = ocr_mode, **kwargs)
                    image_content = image_content.to_markdown() 
                    if kwargs.get("process_tags", False):
                        image_content = self.process_tags(image_content)                  
                    if image_content:
                        if full_text.endswith('\n'):
                            full_text += f"{image_content}\n"
                        else:
                            full_text += f"\n{image_content}\n"
            except:
                tt = time.time()
                traceback.print_exc()
                pix = page.get_pixmap(dpi = 72)
                img_bytes = pix.tobytes("png")
                # 使用BytesIO读取字节流，并使用Pillow打开图像
                pil_image = Image.open(io.BytesIO(img_bytes))                
                image_content = ocrmodel.invoke(pil_image, mode = "base", **kwargs)
                image_content = image_content.to_markdown()                   
                if image_content:
                    if full_text.endswith('\n'):
                        full_text += f"{image_content}\n"
                    else:
                        full_text += f"\n{image_content}\n"
            time_res = time.time() - tt
            time_elapse.append(time_res)
            if parsecallback:
                parseinfo = {
                    "file": file.name,
                    "total_pages": total_page_count,
                    "finish_page": page_number + 1,
                    "pdf_classify": pdf_classify_result.model_dump() if not isinstance(pdf_classify_result, dict) else pdf_classify_result,
                    "page_elapse_time": time_res,
                    "finish_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                parsecallback.on_page_parsed(**parseinfo)
            res.append({
                "pageid": page_number + 1,
                "type": "pdf",
                "content": full_text,
                "images": images,
                "time_elapse": time_elapse
            })
        if parsecallback:
            parseinfo = {
                "file": file.name,
                "total_pages": total_page_count,
                "total_elapse_time": sum(time_elapse),
                "finish_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            parsecallback.on_finished(**parseinfo)
        return res