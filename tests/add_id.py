import h5py
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def main(
    input_h5: Path,
    identitie_path: Path,
):
    with h5py.File(input_h5, "a") as f:
        total_len = len(f["images"])
        identities = f.create_dataset(
            "identities",
            (total_len,),
            dtype=np.int32,
        )
        colums = ["filename", "identity"]
        df = pd.read_csv(identitie_path, header=None, sep=" ", names=colums)
        bar = tqdm(df.iterrows(), total=total_len, desc="Processing images")
        for index, row in bar:
            identities[index] = row["identity"]


if __name__ == "__main__":
    main(
        "data-bin/celeba/CelebAMask-HQ/gender.h5",
        "data-bin/celeba/CelebA-HQ-identity.txt",
    )
