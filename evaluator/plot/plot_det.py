import matplotlib.pyplot as plt
import seaborn as sns

from evaluator.toolkit import *
from pathlib import Path
import evaluator.models.t5_sentinel.t5_get_hidden_states as T5_Full


def get_t5_one_to_rest_full_det(prediction_idx: int, pos_label: str):
    predictions = T5_Full.evaluate_predictions([
        Path("data", "split", "open-web-text", "test-dirty.jsonl"),
        Path("data", "split", "open-gpt-text", "test-dirty.jsonl"),
        Path("data", "split", "open-palm-text", "test-dirty.jsonl"),
        Path("./data/split/open-llama-text/test-dirty.jsonl"),
        Path("./data/split/gpt2-output/test-dirty.jsonl")
    ])
    reformulated_predictions = []
    for entry in predictions:
        p_selected = entry["data"][prediction_idx]
        new_entry = entry.copy()
        new_entry["data"] = np.array([p_selected, 1 - p_selected])
        reformulated_predictions.append(new_entry)
    curve = get_det_binary(reformulated_predictions, pos_label)
    return curve

def plot_t5_full_one_to_rest_dirty():
    curve0 = get_t5_one_to_rest_full_det(0, "openweb")
    curve1 = get_t5_one_to_rest_full_det(1, "chatgpt")
    curve2 = get_t5_one_to_rest_full_det(2, "palm")
    curve3 = get_t5_one_to_rest_full_det(3, "llama")
    curve4 = get_t5_one_to_rest_full_det(4, "gpt2_xl")

    figure: plt.Figure = plt.figure(dpi=200)
    ax: plt.Axes = figure.add_subplot(1, 1, 1)
    # ax.set_prop_cycle('color', sns.color_palette("hls"))
    ax.set_yscale('log')
    ax.set_xscale('log')
    ticks_to_use = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
    ax.set_xticks(ticks_to_use)
    ax.set_yticks(ticks_to_use)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("False Negative Rate")
    ax.plot(curve0[0], curve0[1], label="Human")
    ax.plot(curve1[0], curve1[1], label="GPT3.5")
    ax.plot(curve2[0], curve2[1], label="PaLM")
    ax.plot(curve3[0], curve3[1], label="LLaMA")
    ax.plot(curve4[0], curve4[1], label="GPT2-XL")
    ax.set_xlim(0.001, 1.01)
    ax.set_ylim(0.001, 1.01)
    # ax.set_title("DET Curves for T5-Sentinel for each \nclassification label on one-to-rest classification task")
    ax.grid(visible=True, linestyle="--")
    ax.legend()
    figure.tight_layout()
    figure.savefig(Path("./result/t5_sentinel/det_t5_full_dirty.pdf"))


if __name__ == "__main__":
    TASKS = [
        plot_t5_full_one_to_rest_dirty
    ]

    for task in TASKS:
        print("Executing task: ", task.__name__)
        task()
