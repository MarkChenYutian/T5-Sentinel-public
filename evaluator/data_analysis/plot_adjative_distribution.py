import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches

import nltk
import memoizer
from evaluator.toolkit import *
from tqdm import tqdm
import string


def count_character(s: str, counter: dict) -> None:
    text       = nltk.word_tokenize(s)
    pos_tagged = nltk.pos_tag(text, tagset='universal')
    for c in pos_tagged:
        if c in counter: counter[c] += 1
        else: counter[c] = 1


def argeq(a, b): return a[0] == b[0]


@memoizer.memoize(Path("cache", "dataset_adj_count.pt"), arg_eq=argeq)
def count_dataset(dataset_name: Tp.Literal["human", "chatgpt", "palm", "llama", "gpt2"]) -> dict:
    selected_files = {
        "human": Human_Data, "chatgpt": GPT3_Data, "palm": PaLM_Data, "llama": LLaMA_Data, "gpt2": GPT2_Data
    }[dataset_name]

    counter = {c: 0 for c in string.printable}
    dataset = load_data(selected_files[1:])
    for entry in tqdm(dataset): count_character(entry, counter)

    total_token = sum([counter[k] for k in counter])
    result = {k: counter[k] / total_token for k in counter}
    return result



def get_top_k_chars(counter: dict, k: int) -> list:
    kv_pair = [(counter[k], k) for k in counter]
    kv_pair.sort(key=lambda x: x[0], reverse=True)
    return [entry[1] for entry in kv_pair[:k]]


def filter_dict(counter: dict, keys: list) -> dict:
    resulted_dict = {}
    for k in keys:
        resulted_dict[k] = counter[k] if k in counter else 0
    return resulted_dict


def merge_keys(*arr_keys) -> list:
    set_keys = set()
    for keys in arr_keys: set_keys = set_keys.union(set(keys))
    return list(set_keys)


def sort_keys(counter, keys) -> list:
    key_arr = [(counter[key], key) for key in keys]
    key_arr.sort(key=lambda x: x[0], reverse=True)
    return [entry[1] for entry in key_arr]



def plot_adj_distribution():
    human_counter = count_dataset("human")
    gpt3_counter  = count_dataset("chatgpt")
    palm_counter  = count_dataset("palm")
    llama_counter = count_dataset("llama")
    gpt2_counter  = count_dataset("gpt2")

    select_tok = get_top_k_chars(human_counter, k=40)

    human_counter = filter_dict(human_counter, select_tok)
    gpt3_counter = filter_dict(gpt3_counter, select_tok)
    palm_counter = filter_dict(palm_counter, select_tok)
    llama_counter = filter_dict(llama_counter, select_tok)
    gpt2_counter  = filter_dict(gpt2_counter, select_tok)
    selected_keys = sort_keys(human_counter, select_tok)

    fig, axes = plt.subplots(ncols=1, nrows=5, dpi=200, sharey=True, sharex=True)
    colors = ["#2576b0", "#fc822e", "#349f3c", "#d32f2e", "#9368b9"]
    categories = ["Human", "GPT3.5", "PaLM", "LLaMA", "GPT2"]
    all_data = [human_counter, gpt3_counter, palm_counter, llama_counter, gpt2_counter]

    for idx, (category, counter, color, ax) in enumerate(zip(categories, all_data, colors, axes)):
        values = [counter[k] for k in selected_keys]
        keys   = [k[0] for k in selected_keys]
        # print(selected_keys, values)
        # print(type(selected_keys[0]), type(values[0]), type(color))
        ax.bar(keys, values, color=color)

    # for ax in axes[:-1]: ax.get_xaxis().set_visible(False)
    for ax in axes: ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    handles = [mpatches.Patch(color=c, label=label) for c, label in zip(colors, categories)]
    fig: plt.Figure
    axes: Tp.List[plt.Axes]
    axes[0].legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), handles=handles)
    # axes[-1].set_xlabel(axes[-1].get_xlabel(), rotation=90)
    axes[-1].set_xticklabels(keys, rotation=45)

    fig.tight_layout()
    # axes[1].legend(handles=handles)
    fig.subplots_adjust(wspace=0, hspace=0.2)
    fig.text(0.00, 0.5, 'Frequency', va='center', rotation='vertical')
    fig.text(0.5, 0.01, 'Word classes distribution in OpenLLMText dataset', ha='center')

    fig.savefig("./result/data/dataset_adj_count.pdf")


if __name__ == "__main__":
    TASKS = [
        plot_adj_distribution
    ]
    for task in TASKS:
        print(f"Executing {task.__name__}")
        task()
