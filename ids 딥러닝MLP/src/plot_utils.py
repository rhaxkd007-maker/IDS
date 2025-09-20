import matplotlib.pyplot as plt
from pathlib import Path

def plot_learning_curves(hist_json_path, out_png):
    import json
    with open(hist_json_path, "r") as f:
        h = json.load(f)
    ep = h["epoch"]
    plt.figure()
    plt.plot(ep, h["loss"], label="loss")
    plt.plot(ep, h["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Learning Curve (Loss)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
