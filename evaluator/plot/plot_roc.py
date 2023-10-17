import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc

from evaluator.toolkit import *
from pathlib import Path

from pipeline.lib.import_openai_result import import_openai_prediction_result
from pipeline.lib.import_zerogpt_result import import_zerogpt_prediction_result

import evaluator.models.t5_sentinel.t5_get_hidden_states                   as T5_Full
import evaluator.models.t5_hidden.t5_get_hidden_states       as T5_Hidden


def get_openai_baseline_curve():
    predictions = import_openai_prediction_result()

    reformulated_predictions = []
    for entry in predictions:
        p_selected = entry["data"][0]
        new_entry = entry.copy()
        new_entry["data"] = np.array([p_selected, 1 - p_selected])
        reformulated_predictions.append(new_entry)

    curve = get_roc_binary(reformulated_predictions, "openweb")
    return curve


def get_zerogpt_baseline_curve():
    prediction = import_zerogpt_prediction_result()

    reformulated_predictions = []
    for entry in prediction:
        p_selected = entry["data"][0]
        new_entry = entry.copy()
        new_entry["data"] = np.array([p_selected, 1 - p_selected])
        reformulated_predictions.append(new_entry)

    curve = get_roc_binary(reformulated_predictions, "openweb")
    return curve



def get_t5_one_to_rest_roc_full(file_name, prediction_idx: int, pos_label: str):
    predictions = T5_Full.evaluate_predictions([
        Path("data", "split", "open-web-text", file_name),
        Path("data", "split", "open-gpt-text", file_name),
        Path("data", "split", "open-palm-text", file_name),
        Path("./data/split/open-llama-text/", file_name),
        Path("./data/split/gpt2-output/", file_name)
    ])
    reformulated_predictions = []
    for entry in predictions:
        p_selected = entry["data"][prediction_idx]
        new_entry = entry.copy()
        new_entry["data"] = np.array([p_selected, 1 - p_selected])
        reformulated_predictions.append(new_entry)
    curve = get_roc_binary(reformulated_predictions, pos_label)
    return curve


def get_t5_one_to_rest_roc_hidden(file_name, prediction_idx: int, pos_label: str):
    predictions = T5_Hidden.evaluate_predictions([
        Path("data", "split", "open-web-text"  , file_name),
        Path("data", "split", "open-gpt-text"  , file_name),
        Path("data", "split", "open-palm-text" , file_name),
        Path("data", "split", "open-llama-text", file_name),
        Path("data", "split", "gpt2-output"    , file_name)
    ])
    reformulated_predictions = []
    for entry in predictions:
        p_selected = entry["data"][prediction_idx]
        new_entry = entry.copy()
        new_entry["data"] = np.array([p_selected, 1 - p_selected])
        reformulated_predictions.append(new_entry)
    curve = get_roc_binary(reformulated_predictions, pos_label)
    return curve

def plot_t5_full_one_to_rest():
    curve0 = get_t5_one_to_rest_roc_full("test-dirty.jsonl", 0, "openweb")
    curve1 = get_t5_one_to_rest_roc_full("test-dirty.jsonl", 1, "chatgpt")
    curve2 = get_t5_one_to_rest_roc_full("test-dirty.jsonl", 2, "palm")
    curve3 = get_t5_one_to_rest_roc_full("test-dirty.jsonl", 3, "llama")
    curve4 = get_t5_one_to_rest_roc_full("test-dirty.jsonl", 4, "gpt2_xl")

    print(f"Human   AUC: {auc(curve0[0], curve0[1])}")
    print(f"GPT3.5  AUC: {auc(curve1[0], curve1[1])}")
    print(f"PaLM    AUC: {auc(curve2[0], curve2[1])}")
    print(f"LLaMA   AUC: {auc(curve3[0], curve3[1])}")
    print(f"GPT2-XL AUC: {auc(curve4[0], curve4[1])}")

    figure: plt.Figure = plt.figure(dpi=200)
    ax: plt.Axes = figure.add_subplot(1, 1, 1)
    # ax.set_prop_cycle('color', sns.color_palette("hls"))
    ax.plot(curve0[0], curve0[1], label="Human")
    ax.plot(curve1[0], curve1[1], label="GPT3.5")
    ax.plot(curve2[0], curve2[1], label="PaLM")
    ax.plot(curve3[0], curve3[1], label="LLaMA")
    ax.plot(curve4[0], curve4[1], label="GPT2-XL")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    # ax.set_title("ROC Curves for T5-Sentinel for each \nclassification label on one-to-rest classification task")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(visible=True, linestyle="--")
    ax.legend()
    figure.tight_layout()
    figure.savefig(Path("./result/t5_sentinel/roc_t5_full.pdf"))


def plot_t5_hidden_one_to_rest():
    curve0 = get_t5_one_to_rest_roc_hidden("test-dirty.jsonl", 0, "openweb")
    curve1 = get_t5_one_to_rest_roc_hidden("test-dirty.jsonl", 1, "chatgpt")
    curve2 = get_t5_one_to_rest_roc_hidden("test-dirty.jsonl", 2, "palm")
    curve3 = get_t5_one_to_rest_roc_hidden("test-dirty.jsonl", 3, "llama")
    curve4 = get_t5_one_to_rest_roc_hidden("test-dirty.jsonl", 4, "gpt2_xl")

    print(f"Human   AUC: {auc(curve0[0], curve0[1])}")
    print(f"GPT3.5  AUC: {auc(curve1[0], curve1[1])}")
    print(f"PaLM    AUC: {auc(curve2[0], curve2[1])}")
    print(f"LLaMA   AUC: {auc(curve3[0], curve3[1])}")
    print(f"GPT2-XL AUC: {auc(curve4[0], curve4[1])}")

    figure: plt.Figure = plt.figure(dpi=200)
    ax: plt.Axes = figure.add_subplot(1, 1, 1)
    # ax.set_prop_cycle('color', sns.color_palette("hls"))
    ax.plot(curve0[0], curve0[1], label="Human")
    ax.plot(curve1[0], curve1[1], label="GPT3.5")
    ax.plot(curve2[0], curve2[1], label="PaLM")
    ax.plot(curve3[0], curve3[1], label="LLaMA")
    ax.plot(curve4[0], curve4[1], label="GPT2-XL")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    # ax.set_title("ROC Curves for T5-Sentinel for each \nclassification label on one-to-rest classification task")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(visible=True, linestyle="--")
    ax.legend()
    figure.tight_layout()
    figure.savefig(Path("./result/t5_sentinel/roc_t5_hidden.pdf"))


def plot_t5_full_ablation(pos: int, label: str):
    def implement() -> None:
        curve_0 = get_t5_one_to_rest_roc_full("test-dirty.jsonl", pos, label)
        curve_1 = get_t5_one_to_rest_roc_full("test.variant1.jsonl", pos, label)
        curve_2 = get_t5_one_to_rest_roc_full("test.variant2.jsonl", pos, label)
        curve_3 = get_t5_one_to_rest_roc_full("test.variant3.jsonl", pos, label)
        curve_4 = get_t5_one_to_rest_roc_full("test.variant4.jsonl", pos, label)

        figure: plt.Figure = plt.figure(dpi=200)
        ax: plt.Axes = figure.add_subplot(1, 1, 1)
        ax.set_prop_cycle('color', sns.color_palette("hls"))
        ax.plot(curve_0[0], curve_0[1], label="Original")
        ax.plot(curve_1[0], curve_1[1], label="Remove Newline")
        ax.plot(curve_2[0], curve_2[1], label="Unicode to ASCII")
        ax.plot(curve_3[0], curve_3[1], label="Remove Punctuations")
        ax.plot(curve_4[0], curve_4[1], label="To Lower")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        # ax.set_title(
        #     f"ROC Curves for T5-Sentinel for each \ndifferent sanitization level on one-to-rest classification task ({label})")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.grid(visible=True, linestyle="--")
        ax.legend()
        figure.savefig(Path(f"./result/t5_sentinel/ablation_{label}_roc.pdf"))

    return implement


def plot_t5_compare_with_baseline():
    curve_openai = get_openai_baseline_curve()
    curve_zerogpt = get_zerogpt_baseline_curve()
    curve_t5 = get_t5_one_to_rest_roc_full("test-dirty.jsonl", 0, "openweb")
    curve_hidden = get_t5_one_to_rest_roc_hidden("test-dirty.jsonl", 0, "openweb")

    print(f"OpenAI      AUC: {auc(curve_openai[0], curve_openai[1])}")
    print(f"ZeroGPT     AUC: {auc(curve_zerogpt[0], curve_zerogpt[1])}")
    print(f"T5-Sentinel AUC: {auc(curve_t5[0], curve_t5[1])}")
    print(f"T5-Hidden   AUC: {auc(curve_hidden[0], curve_t5[1])}")

    figure: plt.Figure = plt.figure(dpi=200)
    ax: plt.Axes = figure.add_subplot(1, 1, 1)
    # ax.set_prop_cycle('color', sns.color_palette("hls"))
    ax.plot(curve_openai[0], curve_openai[1], label="OpenAI")
    ax.plot(curve_zerogpt[0], curve_zerogpt[1], label="ZeroGPT")
    ax.plot(curve_t5[0], curve_t5[1], label="T5-Sentinel")
    ax.plot(curve_hidden[0], curve_hidden[1], label="T5-Hidden")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    # ax.set_title(f"ROC Curves for T5-Sentinel on Identifying human")
    ax.grid(visible=True, linestyle="--")
    ax.legend()
    figure.tight_layout()
    figure.savefig(Path(f"./result/t5_sentinel/t5_compare_with_baseline.pdf"))


if __name__ == "__main__":
    TASKS = [
        plot_t5_full_one_to_rest,
        plot_t5_full_ablation(0, "openweb"),
        plot_t5_full_ablation(1, "chatgpt"),
        plot_t5_full_ablation(2, "palm"),
        plot_t5_full_ablation(3, "llama"),
        plot_t5_full_ablation(4, "gpt2_xl"),
        plot_t5_compare_with_baseline
    ]

    for task in TASKS:
        print("Executing task: ", task.__name__)
        task()
