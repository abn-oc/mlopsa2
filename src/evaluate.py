import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow import keras


def main():
    root = Path(__file__).resolve().parents[1]
    model = keras.models.load_model(root / "models/model.h5")
    with np.load(root / "data/processed/test.npz") as data:
        images, labels = data["images"], data["labels"]
    loss, accuracy = model.evaluate(images, labels, verbose=0)
    predictions = model.predict(images, verbose=0).argmax(axis=1)
    matrix = confusion_matrix(labels, predictions, labels=np.arange(10))
    ConfusionMatrixDisplay(matrix, display_labels=np.arange(10)).plot()
    plt.tight_layout()
    plt.savefig(root / "models/confusion_matrix.png")
    plt.close()
    with (root / "metrics.json").open("w") as f:
        json.dump({
            "test_loss": float(loss),
            "test_accuracy": float(accuracy),
            "confusion_matrix": matrix.tolist(),
        }, f, indent=2)


if __name__ == "__main__":
    main()
