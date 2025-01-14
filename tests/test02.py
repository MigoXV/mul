import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def check_png_resolutions(directory: Path):
    # 遍历目录下所有PNG文件
    png_files = list(directory.glob('*.jpg'))
    
    resolutions = set()  # 用来存储所有不同的分辨率

    for file in tqdm(png_files, desc="Checking PNG files", unit="file"):
        try:
            with Image.open(file) as img:
                # 仅访问图片的分辨率，不加载图片内容
                resolutions.add(img.size)
        except Exception as e:
            print(f"Error opening {file}: {e}")
    
    return resolutions

if __name__ == "__main__":
    dir_path = Path("data-bin/celeba/CelebAMask-HQ/CelebA-HQ-img")
    resolutions = check_png_resolutions(dir_path)
    
    print("\n不同的PNG分辨率有:")
    for resolution in sorted(resolutions):
        print(f"{resolution[0]}x{resolution[1]}")
