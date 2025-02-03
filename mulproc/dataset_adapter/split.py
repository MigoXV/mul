from pathlib import Path
from typing import List

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


def save_h5(
    input_h5_file: h5py.File,
    output_h5: Path,
    raw_indices: np.ndarray,
    dataset_names: List,
):
    # total_len = len(raw_indices)
    raw_indices = np.sort(raw_indices)
    with h5py.File(output_h5, "w") as f:
        # 遍历所有数据集，并在输出文件中创建对应的数据集
        for name in dataset_names:
            f.create_dataset(name, data=input_h5_file[name][raw_indices])

        # for indice, raw_indice in enumerate(
        #     tqdm(raw_indices, desc=f"Gathering {output_h5.stem}", leave=False)
        # ):
        #     for name, data in datasets.items():
        #         f[name][indice] = data[raw_indice]


def split(
    raw_h5: Path,
    output_dir: Path,
    train_index: int,
    retain_index: int,
    unseen_index: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(raw_h5, "r") as f:
        # 动态获取所有数据集的名称
        dataset_names = list(f.keys())
        # 获取身份数据
        identity_ds = f["identity"]
        identites = np.array(identity_ds)
        indices = np.arange(len(identites))
        unique_identities = np.unique(identites)
        print(f"唯一身份的数量: {len(unique_identities)}")
        # 根据给定的索引划分数据集
        # 测试集
        test_indices = indices[identites < train_index]
        # 训练集
        train_indices = indices[
            (identites >= train_index) & (identites < unseen_index)
        ]
        # 未见集
        unseen_indices = indices[identites >= unseen_index]
        # 保留集
        retain_indices = indices[
            (identites >= retain_index) & (identites < unseen_index)
        ]
        # 遗忘集
        forget_indices = indices[
            (identites >= train_index) & (identites < retain_index)
        ]
        dataset_indices = {
            "test": test_indices,
            "train": train_indices,
            "unseen": unseen_indices,
            "retain": retain_indices,
            "forget": forget_indices,
        }
        dataset_id_nums = {
            "test": train_index,
            "train": unseen_index - train_index,
            "unseen": len(unique_identities) - unseen_index,
            "retain": unseen_index - retain_index,
            "forget": retain_index - train_index,
        }
        for name, indices in dataset_indices.items():
            save_h5(f, output_dir / f"{name}.h5", indices, dataset_names)
            print(
                f"{name}: {len(indices)} 张图片, {dataset_id_nums[name]} 个身份"
            )
        # save_h5(f,output_dir / "test.h5", test_indices, dataset_names)
        # print(
        #     f"Test: {len(test_indices)} 张图片, {len(np.unique(test_indices))} 个身份"
        # )

        # # save_h5(output_dir / "train.h5", train_indices, datasets)
        # save_h5(f,output_dir / "train.h5", train_indices, dataset_names)
        # print(
        #     f"Train: {len(train_indices)} 张图片, {len(np.unique(train_indices))} 个身份"
        # )

        # # save_h5(output_dir / "unseen.h5", unseen_indices, datasets)
        # save_h5(f,output_dir / "unseen.h5", unseen_indices, dataset_names)
        # print(
        #     f"Unseen: {len(unseen_indices)} 张图片, {len(np.unique(unseen_indices))} 个身份"
        # )
        # # 进一步划分训练集

        # # save_h5(output_dir / "retain.h5", retain_indices, datasets)
        # save_h5(f,output_dir / "retain.h5", retain_indices, dataset_names)
        # print(
        #     f"Retain: {len(retain_indices)} 张图片, {len(np.unique(retain_indices))} 个身份"
        # )

        # # save_h5(output_dir / "forget.h5", forget_indices, datasets)
        # save_h5(f,output_dir / "forget.h5", forget_indices, dataset_names)
        # print(
        #     f"Forget: {len(forget_indices)} 张图片, {len(np.unique(forget_indices))} 个身份"
        # )


if __name__ == "__main__":
    raw_h5 = "data-bin/celeba/celeba.h5"
    output_dir = "data-bin/celeba/split"
    raw_h5 = Path(raw_h5)
    output_dir = Path(output_dir)
    split(raw_h5, output_dir, 190, 1250, 4855)
