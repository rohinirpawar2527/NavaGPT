
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

data = pd.read_csv("experiments/training_history.csv")

plt.figure(figsize=(9, 5))

plt.plot(
    data["step"],
    data["training_loss"],
    marker="o",
    label="Training Loss"
)

plt.plot(
    data["step"],
    data["validation_loss"],
    marker="s",
    label="Validation Loss"
)

plt.title("NavaGPT Training Performance")
plt.xlabel("Training Steps")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)
plt.tight_layout()

Path("experiments").mkdir(exist_ok=True)

plt.savefig(
    "experiments/training_loss.png",
    dpi=300
)

plt.show()

print("Graph saved: experiments/training_loss.png")
