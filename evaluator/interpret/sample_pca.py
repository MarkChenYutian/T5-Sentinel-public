import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from evaluator.plot.plot_pca import pca_analysis

import evaluator.models.t5_sentinel.t5_get_hidden_states as T5_Full


def retrieve_nearest_neighbor_center(points):
    # points - N x 2 ndarray
    center = np.mean(points, axis=0)
    nearest_neighbor_idx = np.argmin(np.linalg.norm(points - center, axis=1))
    return points[nearest_neighbor_idx], nearest_neighbor_idx.item()


def sample_pca_t5_full_center():
    evaluate_paths = [
        Path("data", "split", "open-web-text", "test-dirty.jsonl"),
        Path("data", "split", "open-gpt-text", "test-dirty.jsonl"),
        Path("data", "split", "open-palm-text", "test-dirty.jsonl"),
        Path("./data/split/open-llama-text/test-dirty.jsonl"),
        Path("./data/split/gpt2-output/test-dirty.jsonl"),
    ]

    hiddens = T5_Full.evaluate_hidden_states(evaluate_paths)
    labels = ["openweb", "chatgpt", "palm", "llama", "gpt2_xl"]
    Tarrays = pca_analysis(hiddens, labels, "t5_full_pca")

    fig = plt.figure(dpi=200)
    ax: plt.Axes = fig.add_subplot(1, 1, 1)
    ax.set_prop_cycle('color', sns.color_palette("pastel"))
    ax.set_title("PCA projection of decoder hidden state\nT5-Sentinel")
    for transformed, label in zip(Tarrays, labels):
        ax.scatter(transformed[:, 0], transformed[:, 1], label=label, s=1)

    ax.set_prop_cycle('color', sns.color_palette("dark"))
    f = open("./result/t5_sentinel/pca_sampling_result.txt", "w")
    f.writelines(["This file is dynamically generated from sample_pca.py, do not modify\n"])

    for eval_path, transformed in zip(evaluate_paths, Tarrays):
        cluster_hiddens = T5_Full.evaluate_hiddens_impl(eval_path)
        center, center_idx = retrieve_nearest_neighbor_center(transformed)
        center_uid, center_extra = cluster_hiddens[center_idx]["uid"], cluster_hiddens[center_idx]["extra"]
        ax.scatter(center[0], center[1], marker="*", s=15)
        f.writelines([f"UID: {center_uid} | extra: {center_extra}\n"])

    f.close()
    ax.legend()
    fig.savefig(Path("./result/t5_sentinel/pca_sampling.pdf"))


def sample_pca_t5_full_edge():
    evaluate_paths = [
        Path("data", "split", "open-web-text", "test-dirty.jsonl"),
        Path("data", "split", "open-gpt-text", "test-dirty.jsonl"),
        Path("data", "split", "open-palm-text", "test-dirty.jsonl"),
        Path("./data/split/open-llama-text/test-dirty.jsonl"),
        Path("./data/split/gpt2-output/test-dirty.jsonl"),
    ]

    hiddens = T5_Full.evaluate_hidden_states(evaluate_paths)
    labels = ["openweb", "chatgpt", "palm", "llama", "gpt2_xl"]
    Tarrays = pca_analysis(hiddens, labels, "t5_full_pca")

    fig = plt.figure(dpi=200)
    ax: plt.Axes = fig.add_subplot(1, 1, 1)
    ax.set_prop_cycle('color', sns.color_palette("pastel"))
    ax.set_title("PCA projection of decoder hidden state\nT5-Sentinel")
    for transformed, label in zip(Tarrays, labels):
        random_mask = np.random.randn(*(transformed[:, 0].shape)) < 0.2
        ax.scatter(transformed[random_mask, 0], transformed[random_mask, 1], label=label, s=1)
        # ax.scatter(transformed[:, 0], transformed[:, 1], label=label, s=1)

    f = open("./result/t5_sentinel/pca_sampling_result_edges.txt", "w")
    f.write("This file is dynamically generated from sample_pca.py, do not modify\n")

    top_most = Tarrays[1]  # ChatGPT
    right_most = Tarrays[2]  # PaLMs
    bottom_most = Tarrays[3]  # LLaMA
    left_most   = Tarrays[4]    # GPT2

    left_most_idx   = np.argmin(left_most[:, 0], axis=0)
    right_most_idx  = np.argmax(right_most[:, 0], axis=0)
    top_most_idx    = np.argmax(top_most[:, 1], axis=0)
    bottom_most_idx = np.argmin(bottom_most[:, 1], axis=0)

    ax.set_prop_cycle('color', sns.color_palette("dark"))
    # The order of plotting here matters!
    ax.scatter([], [], marker="*", s=15)
    ax.scatter(top_most[top_most_idx, 0], top_most[top_most_idx, 1], marker="*", s=15)
    ax.scatter(right_most[right_most_idx, 0]  , right_most[right_most_idx, 1] , marker="*", s=15)
    ax.scatter(bottom_most[bottom_most_idx, 0], bottom_most[bottom_most_idx, 1], marker="*", s=15)
    ax.scatter(left_most[left_most_idx, 0], left_most[left_most_idx, 1], marker="*", s=15)

    top_most_preds    = T5_Full.evaluate_predictions([evaluate_paths[1]])
    right_most_preds  = T5_Full.evaluate_predictions([evaluate_paths[2]])
    bottom_most_preds = T5_Full.evaluate_predictions([evaluate_paths[3]])
    left_most_preds   = T5_Full.evaluate_predictions([evaluate_paths[4]])

    top_most_uid, top_most_extra       = top_most_preds[top_most_idx.item()]["uid"], top_most_preds[top_most_idx.item()]["extra"]
    right_most_uid, right_most_extra   = right_most_preds[right_most_idx.item()]["uid"], right_most_preds[right_most_idx.item()]["extra"]
    bottom_most_uid, bottom_most_extra = bottom_most_preds[bottom_most_idx.item()]["uid"], bottom_most_preds[bottom_most_idx.item()]["extra"]
    left_most_uid, left_most_extra     = left_most_preds[left_most_idx.item()]["uid"], left_most_preds[left_most_idx.item()]["extra"]

    f.write(f"TOP: {top_most_uid}, {top_most_extra}\n")
    f.write(f"RIGHT: {right_most_uid}, {right_most_extra}\n")
    f.write(f"BOTTOM: {bottom_most_uid}, {bottom_most_extra}\n")
    f.write(f"LEFT: {left_most_uid}, {left_most_extra}\n")

    f.close()
    ax.legend()
    fig.savefig(Path("./result/t5_sentinel/pca_sampling_edge.pdf"))


if __name__ == "__main__":
    TASKS = [
        # sample_pca_t5_full_center,
        sample_pca_t5_full_edge
    ]

    for task in TASKS:
        print("Executing task: ", task.__name__)
        task()
