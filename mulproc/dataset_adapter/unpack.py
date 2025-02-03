from pathlib import Path

import h5py
import imageio
import numpy as np
from PIL import Image  # 确保正确导入Image模块
from tqdm import tqdm


def unpack_by_identity(
    raw_h5: Path,
    output_dir: Path,
):
    # 将某个h5文件中的图片按照身份进行解包，分别存储到不同的目录中
    output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(raw_h5, "r") as f:
        img_ds = f["image"]
        # filename_ds = f["filename"]
        identities = np.array(f["identity"])
        unique_identities = np.unique(identities)
        for identity in tqdm(unique_identities):
            identity_dir = output_dir / str(identity)
            identity_dir.mkdir(parents=True, exist_ok=True)
            indices = np.where(identities == identity)[0]
            for indice in indices:
                image = img_ds[indice]
                # 放大8倍
                image = Image.fromarray(image)
                image = image.resize((512, 512))
                # filename = filename_ds[indice].decode("utf-8")
                imageio.imwrite(identity_dir / f"{identity}-{indice}.png", image)


if __name__ == "__main__":
    input_h5 = Path("data-bin/celeba/split/test.h5")
    output_dir = Path("data-bin/celeba/split/test")
    unpack_by_identity(input_h5, output_dir)
