import os
import shutil
from pathlib import Path
from ..loggers import logger


allow_model_mirrors = [
    "modelscope",
    "huggingface"
]
model_dowanload_mirror = os.getenv("ANYPARSE_MODEL_MIRROR", "modelscope").lower()
if model_dowanload_mirror not in allow_model_mirrors:
    logger.error(f"Invalid model mirror: {model_dowanload_mirror}, please set one of {allow_model_mirrors}")
    exit(1)


logger.info(f"Download anyparse models from {model_dowanload_mirror}")


def download_anyparse_models(
    model_path: str | os.PathLike = ""
):
    model_id = "anyforge/anyparse-models-hub"
    model_path = Path(model_path)
    
    if model_dowanload_mirror == "modelscope":
        from modelscope import snapshot_download
        source_dir = snapshot_download(
            model_id,
            allow_patterns=f"models/{model_path.name}/*"
        )    
    elif model_dowanload_mirror == "huggingface":
        from huggingface_hub import snapshot_download
        source_dir = snapshot_download(
            model_id,
            allow_patterns=f"models/{model_path.name}/*"
        )  
        print(source_dir)  
    source_dir = Path(source_dir) / f"models/{model_path.name}"
    # 3. 遍历源目录下的第一层内容（包含文件和子文件夹）
    for item in source_dir.iterdir():
        target_item = model_path / item.name
        
        # 如果目标位置已经存在，跳过（防止重复复制报错或覆盖）
        if target_item.exists():
            continue
            
        # 核心操作：根据类型选择复制方式
        if item.is_file():
            # copy2 会保留文件的元数据（如修改时间、权限等）
            shutil.copy2(str(item), str(target_item))
            
        elif item.is_dir():
            # copytree 会递归地复制整个文件夹及其内部的所有内容
            shutil.copytree(str(item), str(target_item))

    print(f"🎉 download ok!")