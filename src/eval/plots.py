from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np


def save_confusion_matrix(
    cm: list[list[int]], labels: list[str], out_path: str
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(arr, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Detection Confusion Matrix")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, str(int(arr[i, j])), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_score_bars(scores: dict[str, float], out_path: str, title: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    names = list(scores.keys())
    vals = [float(scores[k]) for k in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, vals, color=["#4c78a8", "#f58518", "#54a24b"])
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.3)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.02, f"{h:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
