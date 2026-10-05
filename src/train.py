import csv
from pathlib import Path

import numpy as np
import yaml
from tensorflow import keras


def main():
    root = Path(__file__).resolve().parents[1]
    with (root / "params.yaml").open() as f:
        params = yaml.safe_load(f)["train"]
    keras.utils.set_random_seed(params["random_state"])
    with np.load(root / "data/processed/train.npz") as data:
        x_train, y_train = data["images"], data["labels"]
    with np.load(root / "data/processed/val.npz") as data:
        x_val, y_val = data["images"], data["labels"]
    model = keras.Sequential([
        keras.Input(shape=x_train.shape[1:]),
        keras.layers.Flatten(),
        keras.layers.Dense(params["hidden_units"], activation="relu"),
        keras.layers.Dropout(params["dropout"]),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss="sparse_categorical_crossentropy", metrics=["accuracy"],
    )
    history = model.fit(
        x_train, y_train, validation_data=(x_val, y_val),
        epochs=params["epochs"], batch_size=params["batch_size"],
    ).history
    models = root / "models"
    models.mkdir(parents=True, exist_ok=True)
    model.save(models / "model.h5")
    with (models / "history.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", *history])
        writer.writerows(
            (epoch, *values)
            for epoch, values in enumerate(zip(*history.values()), start=1)
        )


if __name__ == "__main__":
    main()
