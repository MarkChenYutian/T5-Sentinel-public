import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path

import evaluator.models.t5_sentinel.t5_get_hidden_states as T5_Full


def pca_analysis(hidden_states, labels, task_id):
    hiddens = [
        [entry["data"][-1] for entry in hidden_states if entry["extra"]["source"] == label]
        for label in labels
    ]
    Harrays = [np.concatenate(hidden, axis=0).reshape((len(hidden), -1)) for hidden in hiddens]
    pca_core = PCA(n_components=2)
    pca_core.fit(np.concatenate(Harrays, axis=0))
    Tarrays = [pca_core.transform(hidden) for hidden in Harrays]
    return Tarrays


def plot_t5_full_pca():
    hiddens = T5_Full.evaluate_hidden_states([
        Path("data", "split", "open-web-text", "test-dirty.jsonl"),
        Path("data", "split", "open-gpt-text", "test-dirty.jsonl"),
        Path("data", "split", "open-palm-text", "test-dirty.jsonl"),
        Path("./data/split/open-llama-text/test-dirty.jsonl"),
        Path("./data/split/gpt2-output/test-dirty.jsonl")
    ])
    labels = ["openweb", "chatgpt", "palm", "llama", "gpt2_xl"]
    Tarrays = pca_analysis(hiddens, labels, "t5_full_pca")
    fig = plt.figure(dpi=200)
    ax: plt.Axes = fig.add_subplot(1, 1, 1)
    ax.set_prop_cycle('color', sns.color_palette("hls"))
    ax.set_title("PCA projection of decoder hidden state\nT5-Sentinel")
    for transformed, label in zip(Tarrays, labels):
        ax.scatter(transformed[:, 0], transformed[:, 1], label=label, s=1)
    ax.legend()
    fig.tight_layout()
    fig.savefig("result/t5_sentinel/" + "pca_projection.pdf")


def plot_t5_full_pca_compare():
    hiddens = T5_Full.evaluate_hidden_states([
        Path("data", "split", "open-web-text", "test-dirty.jsonl"),
        Path("data", "split", "open-gpt-text", "test-dirty.jsonl"),
        Path("data", "split", "open-palm-text", "test-dirty.jsonl"),
        Path("./data/split/open-llama-text/test-dirty.jsonl"),
        Path("./data/split/gpt2-output/test-dirty.jsonl")
    ])
    labels = ["openweb", "chatgpt", "palm", "llama", "gpt2_xl"]
    Tarrays = pca_analysis(hiddens, labels, "t5_full_pca")
    fig = plt.figure(dpi=200)
    ax: plt.Axes = fig.add_subplot(1, 1, 1)
    ax.set_prop_cycle('color', sns.color_palette("pastel"))
    ax.set_title("PCA projection of decoder hidden state\nT5-Sentinel")
    for transformed, label in zip(Tarrays, labels):
        random_mask = np.random.randn(*(transformed[:, 0].shape)) < 0.2
        ax.scatter(transformed[random_mask, 0], transformed[random_mask, 1], s=1)

    hiddens = T5_Full.evaluate_hidden_states([
        Path("data", "split", "open-web-text", "test.variant3.jsonl"),
        Path("data", "split", "open-gpt-text", "test.variant3.jsonl"),
        Path("data", "split", "open-palm-text", "test.variant3.jsonl"),
        Path("data", "split", "open-llama-text", "test.variant3.jsonl"),
        Path("data", "split", "gpt2-output", "test.variant3.jsonl")
    ])
    Tarrays = pca_analysis(hiddens, labels, "t5_full_pca_variant3")
    ax.set_prop_cycle('color', sns.color_palette("dark"))
    for transformed, label in zip(Tarrays, labels):
        random_mask = np.random.randn(*(transformed[:, 0].shape)) < 0.2
        ax.scatter(transformed[random_mask, 0], transformed[random_mask, 1], label=label, s=1)

    ax.legend()
    fig.tight_layout()
    fig.savefig("result/t5_sentinel/" + "pca_projection_compare.pdf")



if __name__ == "__main__":
    TASKS = [
        plot_t5_full_pca,
        plot_t5_full_pca_compare
    ]

    for task in TASKS:
        print("Executing task: ", task.__name__)
        task()
