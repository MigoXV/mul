import os
import random
import shutil
from pathlib import Path

import pandas as pd

# 读取id_label.csv
df = pd.read_csv("data-bin/celeba/id_label.csv")

# 随机选择5个身份
unique_identities = df["identity"].unique()
selected_identities = random.sample(list(unique_identities), 5)

# 创建一个目标目录来存放图片
# output_dir = "selected_images"
output_dir = Path("data-bin/celeba/check_by_id")
output_dir.mkdir(exist_ok=True, parents=True)

# 按身份筛选出对应的记录，并根据filename分组
for identity in selected_identities:
    # 获取当前身份的所有记录
    identity_df = df[df["identity"] == identity]

    # 为该身份创建一个文件夹
    identity_folder = Path(output_dir) / f"identity_{identity}"
    identity_folder.mkdir(exist_ok=True, parents=True)

    # 假设图片存储在`filename`列中，我们要将这些图片移动到对应文件夹
    for _, row in identity_df.iterrows():
        filename = row["filename"]
        image_path = Path("data-bin/celeba/CelebA-HQ-img") / filename
        # 假设图片文件存放在`images/`目录下，你可以修改为你实际的图片存放路径
        # image_path = os.path.join("images", filename)
        # image_path = Path("data-bin/celeba/check_by_id") / filename

        # 判断图片文件是否存在
        if os.path.exists(image_path):
            # 将图片移动到对应的身份文件夹
            # shutil.copy(image_path, os.path.join(identity_folder, filename))
            shutil.copy(image_path, Path(identity_folder) / filename)
        else:
            print(f"Warning: {image_path} does not exist!")

print("Images have been successfully copied to the identity folders.")
