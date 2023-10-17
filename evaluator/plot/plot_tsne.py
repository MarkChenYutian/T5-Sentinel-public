import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path

import evaluator.models.t5_sentinel.t5_get_hidden_states as T5_Full
import evaluator.models.t5_hidden.t5_get_hidden_states as T5_Hidden


def tsne_analysis(hidden_states, labels, perplexity):
    hiddens = [
        [entry["data"][-1] for entry in hidden_states if entry["extra"]["source"] == label]
        for label in labels
    ]
    Harrays = [np.concatenate(hidden, axis=0).reshape((len(hidden), -1)) for hidden in hiddens]
    X_original = np.concatenate(Harrays, axis=0)

    pca_core = PCA(n_components=50)
    tsne_core = TSNE(n_components=2, perplexity=perplexity, verbose=1)

    pca_core.fit(X_original)
    X_pca = pca_core.transform(X_original)
    X_tsne = tsne_core.fit_transform(X_pca)

    separate_tsne = []
    accum = 0
    for h in hiddens:
        separate_tsne.append(X_tsne[accum:accum+len(h)])
        accum += len(h)

    return separate_tsne


def plot_t5_full_tsne():
    hiddens = T5_Full.evaluate_hidden_states([
        Path("data", "split", "open-web-text", "test-dirty.jsonl"),
        Path("data", "split", "open-gpt-text", "test-dirty.jsonl"),
        Path("data", "split", "open-palm-text", "test-dirty.jsonl"),
        Path("./data/split/open-llama-text/test-dirty.jsonl"),
        Path("./data/split/gpt2-output/test-dirty.jsonl")
    ])
    labels = ["openweb", "chatgpt", "palm", "llama", "gpt2_xl"]
    display_labels = ["Human", "GPT3.5", "PaLM", "LLaMA", "GPT2-XL"]
    # perplexities = [5, 10, 25, 50, 75, 100]
    perplexities = [100]
    for perplexity in perplexities:
        separate_tsne = tsne_analysis(hiddens, labels, perplexity)

        fig = plt.figure(dpi=200)
        ax: plt.Axes = fig.add_subplot(1, 1, 1)
        # ax.set_prop_cycle('color', sns.color_palette("hls"))
        # ax.set_title(f"t-SNE Plot on Hidden State of T5-Sentinel\nwith perplexity {perplexity}")
        for transformed, label in zip(separate_tsne, display_labels):
            random_mask = np.random.randn(*(transformed[:, 0].shape)) < 0.2
            ax.scatter(transformed[random_mask, 0], transformed[random_mask, 1], s=1)

        colors = ["#2576b0", "#fc822e", "#349f3c", "#d32f2e", "#9368b9"]
        handles = [
            mpatches.Patch(color=c, label=label) for c, label in zip(colors, display_labels)
        ]
        ax.legend(handles=handles)
        fig.tight_layout()
        fig.savefig("result/t5_sentinel/" + "tsne_" + str(perplexity) + ".pdf")


def plot_t5_hidden_tsne():
    hiddens = T5_Hidden.evaluate_hidden_states([
        Path("data", "split", "open-web-text", "test-dirty.jsonl"),
        Path("data", "split", "open-gpt-text", "test-dirty.jsonl"),
        Path("data", "split", "open-palm-text", "test-dirty.jsonl"),
        Path("./data/split/open-llama-text/test-dirty.jsonl"),
        Path("./data/split/gpt2-output/test-dirty.jsonl")
    ])
    labels = ["openweb", "chatgpt", "palm", "llama", "gpt2_xl"]
    display_labels = ["Human", "GPT3.5", "PaLM", "LLaMA", "GPT2-XL"]
    perplexities = [5, 10, 25, 50, 75, 100]
    # perplexities = [100]
    for perplexity in perplexities:
        separate_tsne = tsne_analysis(hiddens, labels, perplexity)

        fig = plt.figure(dpi=200)
        ax: plt.Axes = fig.add_subplot(1, 1, 1)
        # ax.set_prop_cycle('color', sns.color_palette("hls"))
        # ax.set_title(f"t-SNE Plot on Hidden State of T5-Sentinel\nwith perplexity {perplexity}")
        for transformed, label in zip(separate_tsne, display_labels):
            random_mask = np.random.randn(*(transformed[:, 0].shape)) < 0.2
            ax.scatter(transformed[random_mask, 0], transformed[random_mask, 1], s=1)

        colors = ["#2576b0", "#fc822e", "#349f3c", "#d32f2e", "#9368b9"]
        handles = [
            mpatches.Patch(color=c, label=label) for c, label in zip(colors, display_labels)
        ]
        ax.legend(handles=handles)
        fig.tight_layout()
        fig.savefig("result/hidden/" + "tsne_" + str(perplexity) + ".pdf")

if __name__ == "__main__":
    TASKS = [
        plot_t5_full_tsne,
        plot_t5_hidden_tsne
    ]

    for task in TASKS:
        print("Executing task: ", task.__name__)
        task()

