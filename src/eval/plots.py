from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


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


def save_histogram(
    values: list[float] | np.ndarray,
    out_path: str,
    title: str,
    xlabel: str,
    bins: int = 30,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    arr = np.asarray(values, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(arr, bins=bins, edgecolor="black", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_tsne_scatter(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    out_path: str,
    max_points: int = 4000,
    random_state: int = 42,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    work = df.copy()
    work = work.dropna(subset=feature_cols + [label_col])
    if work.empty:
        return

    if len(work) > max_points:
        work = work.sample(n=max_points, random_state=random_state)

    x = work[feature_cols].astype(float).values
    x = StandardScaler().fit_transform(x)

    tsne = TSNE(
        n_components=2,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
        perplexity=30,
    )
    emb = tsne.fit_transform(x)
    work["x"] = emb[:, 0]
    work["y"] = emb[:, 1]

    color_map = {"benign": "green", "suspicious": "orange", "attack_like": "red"}

    fig, ax = plt.subplots(figsize=(8, 6))
    for lbl, grp in work.groupby(label_col):
        ax.scatter(
            grp["x"],
            grp["y"],
            s=10,
            alpha=0.65,
            label=str(lbl),
            color=color_map.get(str(lbl), "gray"),
        )
    ax.set_title("t-SNE Scatter (Session Features)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
