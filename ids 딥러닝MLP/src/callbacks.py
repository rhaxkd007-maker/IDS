from pathlib import Path
import numpy as np
import json
from tensorflow.keras.callbacks import Callback

class EpochHistory(Callback):
    def __init__(self, out_dir: Path):
        super().__init__()
        self.out_dir = Path(out_dir)
        self.hist = {"epoch": [], "loss": [], "val_loss": [], "acc": [], "val_acc": []}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.hist["epoch"].append(int(epoch))
        self.hist["loss"].append(float(logs.get("loss", float("nan"))))
        self.hist["val_loss"].append(float(logs.get("val_loss", float("nan"))))
        self.hist["acc"].append(float(logs.get("accuracy", float("nan"))))
        self.hist["val_acc"].append(float(logs.get("val_accuracy", float("nan"))))
        with open(self.out_dir / "epoch_history.json", "w") as f:
            json.dump(self.hist, f, indent=2)
