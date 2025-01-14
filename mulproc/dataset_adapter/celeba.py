from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import PIL
import PIL.Image
from tqdm import tqdm


def adapter(
    image_dir: Path,
    label_csv_path: Path,
    output_h5: Path,
):
    df = pd.read_csv(label_csv_path)
    # 切片只留文件名和性别
    # male_df = df.iloc[:, ]
    total_len = len(df)
    with h5py.File(output_h5, "w") as f:
        # 创建两个数据集
        images_ds = f.create_dataset(
            "images",
            (total_len, 128, 128, 3),
            dtype="uint8",
        )
        label_ds = f.create_dataset("is_male", (total_len,), dtype=bool)
        identity_ds = f.create_dataset("identity", (total_len,), dtype=np.int32)
        # 迭代
        bar = tqdm(df.iterrows(), total=total_len, desc="Processing images")
        for index, row in bar:
            image = PIL.Image.open(image_dir / row["Filename"])
            image = image.resize((128, 128))
            images_ds[index] = image
            label_ds[index] == True if row["Male"] == 1 else False
            identity_ds[index] = row["Identity"]


if __name__ == "__main__":
    image_dir = Path("data-bin/celeba/CelebA-HQ-img")
    label_csv_path = Path("data-bin/celeba/id_label.csv")
    output_h5 = Path("data-bin/celeba/gender.h5")
    adapter(image_dir, label_csv_path, output_h5)
