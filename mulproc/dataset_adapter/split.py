from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import PIL
import PIL.Image
from tqdm import tqdm


def save_h5(
    output_h5: Path,
    raw_indices: np.ndarray,
    image_ds: h5py.Dataset,
    label_ds: h5py.Dataset,
    identity_ds: h5py.Dataset,
):
    with h5py.File(output_h5, "w") as f:
        total_len = len(raw_indices)
        output_images_ds = f.create_dataset(
            "images",
            (total_len, 128, 128, 3),
            dtype="uint8",
        )
        output_label_ds = f.create_dataset("is_male", (total_len,), dtype=bool)
        output_identity_ds = f.create_dataset("identity", (total_len,), dtype=np.int32)
        for indice, raw_indice in enumerate(
            tqdm(raw_indices, desc=f"Gathering {output_h5.stem}", leave=False)
        ):
            output_images_ds[indice] = image_ds[raw_indice]
            output_label_ds[indice] = label_ds[raw_indice]
            output_identity_ds[indice] = identity_ds[raw_indice]


def split(
    raw_h5: Path,
    output_dir: Path,
    train_index: int,
    retain_index: int,
    unseen_index: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(raw_h5, "r") as f:
        identity_ds = f["identity"]
        image_ds = f["images"]
        label_ds = f["is_male"]
        total_len = len(identity_ds)
        # 把identity_ds转换成numpy array
        identites = np.zeros(total_len, dtype=np.int32)
        for index, identity in enumerate(identity_ds):
            identites[index] = identity
        # 找到所有的identity
        unique_identities = np.unique(identites)
        print(f"number of unique identities: {len(unique_identities)}")
        # 按照顺序划分数据集
        # 测试集，训练集（进一步分为遗忘集和保留集），未见集（本质上是专用于测试遗忘效果的测试集）
        # 测试集
        test_indices = identites[identites < train_index]
        save_h5(output_dir / "test.h5", test_indices, image_ds, label_ds, identity_ds)
        print(
            f"Test:{len(test_indices)} pics, {len(np.unique(test_indices))} identities"
        )
        # 训练集
        train_indices = identites[
            (identites >= train_index) & (identites < unseen_index)
        ]
        save_h5(output_dir / "train.h5", train_indices, image_ds, label_ds, identity_ds)
        print(
            f"Train:{len(train_indices)} pics, {len(np.unique(train_indices))} identities"
        )
        # 未见集
        unseen_indices = identites[identites >= unseen_index]
        save_h5(
            output_dir / "unseen.h5", unseen_indices, image_ds, label_ds, identity_ds
        )
        print(
            f"Unseen:{len(unseen_indices)} pics, {len(np.unique(unseen_indices))} identities"
        )
        # 进一步划分训练集
        # 保留集
        retain_indices = identites[
            (identites >= retain_index) & (identites < unseen_index)
        ]
        save_h5(
            output_dir / "retain.h5", retain_indices, image_ds, label_ds, identity_ds
        )
        print(
            f"Retain:{len(retain_indices)} pics, {len(np.unique(retain_indices))} identities"
        )
        # 遗忘集
        forget_indices = identites[
            (identites >= train_index) & (identites < retain_index)
        ]
        save_h5(
            output_dir / "forget.h5", forget_indices, image_ds, label_ds, identity_ds
        )
        print(
            f"Forget:{len(forget_indices)} pics, {len(np.unique(forget_indices))} identities"
        )


if __name__ == "__main__":
    raw_h5 = "data-bin/celeba/gender.h5"
    output_dir = "data-bin/celeba/split"
    raw_h5 = Path(raw_h5)
    output_dir = Path(output_dir)
    split(raw_h5, output_dir, 190, 1250, 4855)
