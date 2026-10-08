"""Plot the training curves recorded in the outputs of the original notebook.

The original notebook (Hot_dog_Project.ipynb) trained a CNN on Food-101 and printed the
Keras training log. Re-training needs a 4.65 GiB dataset download and hours of CPU time, so this
script only *parses the saved log* from the notebook's output; it does not train anything.
"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
nb = json.loads((HERE / "Hot_dog_Project.ipynb").read_text(encoding="utf-8"))

log = ""
for cell in nb["cells"]:
    for out in cell.get("outputs", []):
        text = "".join(out.get("text", []))
        if "Epoch 1/50" in text:
            log = text

pattern = re.compile(
    r"Epoch (\d+)/\d+.*?loss: ([\d.]+) - accuracy: ([\d.]+) - val_loss: ([\d.]+) - val_accuracy: ([\d.]+)",
    re.S,
)
rows = [tuple(map(float, m)) for m in pattern.findall(log)]
if not rows:
    raise SystemExit("Training log not found in the notebook output.")

epochs, loss, acc, val_loss, val_acc = zip(*rows)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(epochs, loss, label="train"); axes[0].plot(epochs, val_loss, label="validation")
axes[0].set(title="Loss", xlabel="epoch"); axes[0].legend()
axes[1].plot(epochs, acc, label="train"); axes[1].plot(epochs, val_acc, label="validation")
axes[1].set(title="Accuracy", xlabel="epoch"); axes[1].axhline(0.5, color="gray", ls="--", lw=1, label="chance (balanced data)")
axes[1].legend()
fig.suptitle("Hot dog CNN: training log recorded in the original notebook")
fig.tight_layout()
fig.savefig(HERE / "training_curves.png", dpi=130)
print(f"{len(rows)} epochs plotted | last: train acc {acc[-1]:.3f}, val acc {val_acc[-1]:.3f}, best val acc {max(val_acc):.3f}")
