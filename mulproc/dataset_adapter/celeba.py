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
        image_ds = f.create_dataset(
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
            image_ds[index] = image
            label_ds[index] == True if row["Male"] == 1 else False
            identity_ds[index] = row["Identity"]


def full_label_celeba(
    input_h5: Path,
    input_csv: Path,
    output_h5: Path,
):
    # 读取 CSV 文件
    df = pd.read_csv(input_csv)
    # 获取csv文件的列名
    label_names = df.columns
    label_names = label_names[2:]
    label_data_list = []
    for label_name in label_names:
        label_data = df[label_name].values
        label_data = label_data == 1
        label_data_list.append(label_data)
    
    with h5py.File(input_h5, "r") as img_h5, h5py.File(output_h5, "w") as ds_h5:
        # image_ids = img_h5["images"]
        ds_h5.create_dataset("image", data=img_h5["images"])
        ds_h5.create_dataset("identity", data=img_h5["identity"])
        # id_ids = img_h5["identity"]
        for label_name,label_data in zip(label_names,label_data_list):
            ds_h5.create_dataset(label_name, data=label_data)


if __name__ == "__main__":
    # image_dir = Path("data-bin/celeba/CelebA-HQ-img")
    # label_csv_path = Path("data-bin/celeba/id_label.csv")
    # output_h5 = Path("data-bin/celeba/gender.h5")
    # adapter(image_dir, label_csv_path, output_h5)
    input_h5 = "data-bin/celeba/gender.h5"
    input_csv = "data-bin/celeba/id_label.csv"
    output_h5 = "data-bin/celeba/celeba.h5"
    input_h5, input_csv, output_h5 = Path(input_h5), Path(input_csv), Path(output_h5)
    full_label_celeba(input_h5, input_csv, output_h5)
