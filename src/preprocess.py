from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split


def main():
    root = Path(__file__).resolve().parents[1]
    with (root / "params.yaml").open() as f:
        params = yaml.safe_load(f)["preprocess"]
    with np.load(root / "data/raw/train.npz") as data:
        images = data["images"].astype("float32") / 255.0
        labels = data["labels"]
    x_train, x_val, y_train, y_val = train_test_split(
        images, labels, test_size=params["validation_size"],
        random_state=params["random_state"], stratify=labels,
    )
    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(processed / "train.npz", images=x_train, labels=y_train)
    np.savez_compressed(processed / "val.npz", images=x_val, labels=y_val)
    with np.load(root / "data/raw/test.npz") as data:
        np.savez_compressed(
            processed / "test.npz",
            images=data["images"].astype("float32") / 255.0,
            labels=data["labels"],
        )


if __name__ == "__main__":
    main()
