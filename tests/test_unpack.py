import h5py
from PIL import Image
import numpy as np
from pathlib import Path

def unpack_images(h5_file: Path, output_dir: Path, num_images: int = 20):
    # 打开 H5 文件
    with h5py.File(h5_file, "r") as f:
        images = f["images"][:num_images]  # 获取前 20 张图片
        is_male = f["is_male"][:num_images]  # 获取对应的性别标签

        # 逐张保存图片
        for i in range(num_images):
            # 读取图像数据并转换为图片格式
            image_data = images[i]
            img = Image.fromarray(image_data)
            
            # 根据 is_male 标记命名
            gender = "male" if is_male[i] else "female"
            img_name = f"{i}_{gender}.jpg"
            img.save(output_dir / img_name)
            print(f"Saved: {img_name}")

if __name__ == "__main__":
    h5_file = Path("data-bin/celeba/CelebAMask-HQ/gender.h5")
    output_dir = Path("data-bin/celeba/output_images")
    output_dir.mkdir(parents=True, exist_ok=True)  # 创建输出目录
    unpack_images(h5_file, output_dir)
