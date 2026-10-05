from pathlib import Path

import numpy as np
from tensorflow import keras


def main():
    raw = Path(__file__).resolve().parents[1] / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    np.savez_compressed(raw / "train.npz", images=x_train, labels=y_train)
    np.savez_compressed(raw / "test.npz", images=x_test, labels=y_test)


if __name__ == "__main__":
    main()
