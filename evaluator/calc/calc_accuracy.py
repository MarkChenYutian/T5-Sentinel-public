from pathlib import Path
from sklearn.metrics import auc

import evaluator.models.t5_hidden.t5_get_hidden_states as T5_hidden
import evaluator.models.t5_sentinel.t5_get_hidden_states as T5_full

from evaluator.toolkit import *
from pipeline.lib.import_openai_result import import_openai_prediction_result, import_openai_prediction_result_impl
from pipeline.lib.import_zerogpt_result import import_zerogpt_prediction_result, import_zerogpt_prediction_result_impl


def calc_t5_specific_class_hidden(prediction_idx: int, positive_class: str):
    evaluate_paths = [
        Path("./data/split/open-web-text/test-dirty.jsonl"),
        Path("./data/split/open-gpt-text/test-dirty.jsonl"),
        Path("./data/split/open-palm-text/test-dirty.jsonl"),
        Path("./data/split/open-llama-text/test-dirty.jsonl"),
        Path("./data/split/gpt2-output/test-dirty.jsonl")
    ]
    predictions = T5_hidden.evaluate_predictions(evaluate_paths)

    reformulated_predictions = []
    for entry in predictions:
        p_selected = entry["data"][prediction_idx]
        new_entry = entry.copy()
        new_entry["data"] = np.array([p_selected, 1 - p_selected])
        reformulated_predictions.append(new_entry)

    return quick_statistics_binary(reformulated_predictions, positive_class, threshold=0.5)



def calc_t5_specific_class_full(prediction_idx: int, positive_class: str):
    evaluate_paths = [
        Path("./data/split/open-web-text/test-dirty.jsonl"),
        Path("./data/split/open-gpt-text/test-dirty.jsonl"),
        Path("./data/split/open-palm-text/test-dirty.jsonl"),
        Path("./data/split/open-llama-text/test-dirty.jsonl"),
        Path("./data/split/gpt2-output/test-dirty.jsonl")
    ]
    predictions = T5_full.evaluate_predictions(evaluate_paths)

    reformulated_predictions = []
    for entry in predictions:
        p_selected = entry["data"][prediction_idx]
        new_entry = entry.copy()
        new_entry["data"] = np.array([p_selected, 1 - p_selected])
        reformulated_predictions.append(new_entry)

    return quick_statistics_binary(reformulated_predictions, positive_class, threshold=0.5)


def calc_t5_full_statistics(digit=3):
    categories = ["openweb", "chatgpt", "palm", "llama", "gpt2_xl"]
    recalls, precisions, f1s = [], [], []

    for cate_idx, category in enumerate(categories):
        stat = calc_t5_specific_class_full(cate_idx, category)
        print(f"{category} v. rest statistics")
        print(f"TPR: {round(tpr(*stat), digit)},\tTNR: {round(tnr(*stat), digit)}, "
              f"FPR: {round(fpr(*stat), digit)},\tFNR: {round(fnr(*stat), digit)} ")
        print(f"Acc: {round(acc(*stat), digit)}")
        print(f"F1 : {round(f1(*stat), digit)}")
        print(f"Recall: {round(recall(*stat), digit)},\tPrecision: {round(precision(*stat), digit)}")

        recalls.append(recall(*stat))
        precisions.append(precision(*stat))
        f1s.append(f1(*stat))

    print("\nOverall average score:")
    print(f"F1: {round(sum(f1s) / len(f1s), digit)}")
    print(f"Recall: {round(sum(recalls) / len(recalls), digit)},\tPrecision: {round(sum(precisions) / len(precisions), digit)}")


def calc_t5_hidden_statistics(digit=3):
    categories = ["openweb", "chatgpt", "palm", "llama", "gpt2_xl"]
    recalls, precisions, f1s = [], [], []

    for cate_idx, category in enumerate(categories):
        stat = calc_t5_specific_class_hidden(cate_idx, category)
        print(f"{category} v. rest statistics")
        print(f"TPR: {round(tpr(*stat), digit)},\tTNR: {round(tnr(*stat), digit)}, "
              f"FPR: {round(fpr(*stat), digit)},\tFNR: {round(fnr(*stat), digit)} ")
        print(f"Acc: {round(acc(*stat), digit)}")
        print(f"F1 : {round(f1(*stat), digit)}")
        print(f"Recall: {round(recall(*stat), digit)},\tPrecision: {round(precision(*stat), digit)}")

        recalls.append(recall(*stat))
        precisions.append(precision(*stat))
        f1s.append(f1(*stat))

    print("\nOverall average score:")
    print(f"F1: {round(sum(f1s) / len(f1s), digit)}")
    print(f"Recall: {round(sum(recalls) / len(recalls), digit)},\tPrecision: {round(sum(precisions) / len(precisions), digit)}")


def calc_openai_baseline_statistics(digit=3):
    prediction = import_openai_prediction_result()
    reformulated_predictions = []
    for entry in prediction:
        p_selected = entry["data"][0]
        new_entry = entry.copy()
        new_entry["data"] = np.array([p_selected, 1 - p_selected])
        reformulated_predictions.append(new_entry)

    stat = quick_statistics_binary(reformulated_predictions, pos_label="openweb", threshold=0.5)
    curve = get_roc_binary(reformulated_predictions, pos_label="openweb")

    print(f"OpenAI classifier human-to-generated statistics")
    print(f"TPR: {round(tpr(*stat), digit)},\tTNR: {round(tnr(*stat), digit)}, "
          f"FPR: {round(fpr(*stat), digit)},\tFNR: {round(fnr(*stat), digit)} ")
    print(f"Acc: {round(acc(*stat), digit)}")
    print(f"F1 : {round(f1(*stat), digit)}")
    print(f"Recall: {round(recall(*stat), digit)},\tPrecision: {round(precision(*stat), digit)}")
    print(f"AUROC: {round(auc(curve[0], curve[1]), digit)}")


def calc_zerogpt_baseline_statistics(digit=3):
    prediction = import_zerogpt_prediction_result()

    reformulated_predictions = []
    for entry in prediction:
        p_selected = entry["data"][0]
        new_entry = entry.copy()
        new_entry["data"] = np.array([p_selected, 1 - p_selected])
        reformulated_predictions.append(new_entry)

    stat = quick_statistics_binary(reformulated_predictions, pos_label="openweb", threshold=0.5)
    curve = get_roc_binary(reformulated_predictions, pos_label="openweb")

    print(f"ZeroGPT classifier human-to-generated statistics")
    print(f"TPR: {round(tpr(*stat), digit)},\tTNR: {round(tnr(*stat), digit)}, "
          f"FPR: {round(fpr(*stat), digit)},\tFNR: {round(fnr(*stat), digit)} ")
    print(f"Acc: {round(acc(*stat), digit)}")
    print(f"F1 : {round(f1(*stat), digit)}")
    print(f"Recall: {round(recall(*stat), digit)},\tPrecision: {round(precision(*stat), digit)}")
    print(f"AUROC: {round(auc(curve[0], curve[1]), digit)}")


def calc_zerogpt_baseline_statistics_detail(path: Path):
    def implementation(digit=3):
        prediction0 = import_zerogpt_prediction_result_impl(path)
        prediction1 = import_zerogpt_prediction_result_impl(Path(
            "data", "baselines", "zerogpt_classifier_output", "open-web-text.jsonl"
        ))
        prediction = prediction0 + prediction1

        # reformulated_predictions = []
        # for entry in prediction:
        #     p_selected = entry["data"][0]
        #     new_entry = entry.copy()
        #     new_entry["data"] = np.array([p_selected, 1 - p_selected])
        #     reformulated_predictions.append(new_entry)

        stat = quick_statistics_binary(prediction, pos_label="openweb", threshold=0.5)
        curve = get_roc_binary(prediction, pos_label="openweb")

        print(f"ZeroGPT classifier human-to-generated statistics")
        print(path)
        print(f"TPR: {round(tpr(*stat), digit)},\tTNR: {round(tnr(*stat), digit)}, "
              f"FPR: {round(fpr(*stat), digit)},\tFNR: {round(fnr(*stat), digit)} ")
        print(f"Acc: {round(acc(*stat), digit)}")
        print(f"F1 : {round(f1(*stat), digit)}")
        print(f"Recall: {round(recall(*stat), digit)},\tPrecision: {round(precision(*stat), digit)}")
        print(f"AUROC: {round(auc(curve[0], curve[1]), digit)}")
    return implementation


def calc_openai_baseline_statistics_detail(path: Path):
    def implementation(digit=3):
        prediction0 = import_openai_prediction_result_impl(path)
        prediction1 = import_openai_prediction_result_impl(Path(
            "data", "baselines", "openai_classifier_output", "open-web-text.jsonl"
        ))
        prediction = prediction0 + prediction1

        # reformulated_predictions = []
        # for entry in prediction:
        #     p_selected = entry["data"][1]
        #     new_entry = entry.copy()
        #     new_entry["data"] = np.array([p_selected, 1 - p_selected])
        #     reformulated_predictions.append(new_entry)

        stat = quick_statistics_binary(prediction, pos_label="openweb", threshold=0.5)
        curve = get_roc_binary(prediction, pos_label="openweb")

        print(f"OpenAI classifier human-to-generated statistics")
        print(path)
        print(f"TPR: {round(tpr(*stat), digit)},\tTNR: {round(tnr(*stat), digit)}, "
              f"FPR: {round(fpr(*stat), digit)},\tFNR: {round(fnr(*stat), digit)} ")
        print(f"Acc: {round(acc(*stat), digit)}")
        print(f"F1 : {round(f1(*stat), digit)}")
        print(f"Recall: {round(recall(*stat), digit)},\tPrecision: {round(precision(*stat), digit)}")
        print(f"AUROC: {round(auc(curve[0], curve[1]), digit)}")

    return implementation


def calc_t5_full_statistics_detail(file_name: str, pos_label: str, pos_index: int):
    def implementation(digit=3):
        prediction = T5_full.evaluate_predictions([
            Path("data", "split", "open-web-text", file_name),
            Path("data", "split", "open-gpt-text", file_name),
            Path("data", "split", "open-palm-text", file_name),
            Path("data", "split", "open-llama-text", file_name),
            Path("data", "split", "gpt2-output", file_name)
        ])
        reformulated_predictions = []
        for entry in prediction:
            p_selected = entry["data"][pos_index]
            new_entry = entry.copy()
            new_entry["data"] = np.array([p_selected, 1 - p_selected])
            reformulated_predictions.append(new_entry)

        stat = quick_statistics_binary(reformulated_predictions, pos_label, threshold=0.5)
        curve = get_roc_binary(reformulated_predictions, pos_label)

        print(f"T5-Sentinel (ours) classifier statistics")
        print(f"{pos_label}-to-rest @ {file_name}")
        print(f"TPR: {round(tpr(*stat), digit)},\tTNR: {round(tnr(*stat), digit)}, "
              f"FPR: {round(fpr(*stat), digit)},\tFNR: {round(fnr(*stat), digit)} ")
        print(f"Acc: {round(acc(*stat), digit)}")
        print(f"F1 : {round(f1(*stat), digit)}")
        print(f"Recall: {round(recall(*stat), digit)},\tPrecision: {round(precision(*stat), digit)}")
        print(f"AUROC: {round(auc(curve[0], curve[1]), digit)}")

        with open("temp_result.txt", "a") as f:
            f.write(f" & {str(round(auc(curve[0], curve[1]), digit))[1:]} & {str(round(f1(*stat), digit))[1:]}")

    return implementation


def calc_t5_hidden_statistics_detail_full(file_name: str, pos_label: str, pos_index: int):
    def implementation(digit=3):
        prediction = T5_hidden.evaluate_predictions([
            Path("data", "split", "open-web-text", file_name),
            Path("data", "split", "open-gpt-text", file_name),
            Path("data", "split", "open-palm-text", file_name),
            Path("data", "split", "open-llama-text", file_name),
            Path("data", "split", "gpt2-output", file_name)
        ])
        reformulated_predictions = []
        for entry in prediction:
            p_selected = entry["data"][pos_index]
            new_entry = entry.copy()
            new_entry["data"] = np.array([p_selected, 1 - p_selected])
            reformulated_predictions.append(new_entry)

        stat = quick_statistics_binary(reformulated_predictions, pos_label, threshold=0.5)
        curve = get_roc_binary(reformulated_predictions, pos_label)

        print(f"T5-Sentinel (ours) classifier statistics")
        print(f"{pos_label}-to-rest @ {file_name}")
        print(f"TPR: {round(tpr(*stat), digit)},\tTNR: {round(tnr(*stat), digit)}, "
              f"FPR: {round(fpr(*stat), digit)},\tFNR: {round(fnr(*stat), digit)} ")
        print(f"Acc: {round(acc(*stat), digit)}")
        print(f"F1 : {round(f1(*stat), digit)}")
        print(f"Recall: {round(recall(*stat), digit)},\tPrecision: {round(precision(*stat), digit)}")
        print(f"AUROC: {round(auc(curve[0], curve[1]), digit)}")

        with open("temp_result.txt", "a") as f:
            f.write(f" & {str(round(auc(curve[0], curve[1]), digit))[1:]} & {str(round(f1(*stat), digit))[1:]}")

    return implementation


def calc_t5_hidden_statistics_detail(path: Path):
    def implementation(digit=3):
        prediction = T5_hidden.evaluate_predictions([
            path,
            Path("data", "split", "open-web-text", "test-dirty.jsonl")
        ])
        reformulated_predictions = []
        for entry in prediction:
            p_selected = entry["data"][0]
            new_entry = entry.copy()
            new_entry["data"] = np.array([p_selected, 1 - p_selected])
            reformulated_predictions.append(new_entry)

        stat = quick_statistics_binary(reformulated_predictions, "openweb", threshold=0.5)
        curve = get_roc_binary(reformulated_predictions, pos_label="openweb")

        print(f"T5-Sentinel (ours) classifier human-to-generated statistics")
        print(str(path))
        print(f"TPR: {round(tpr(*stat), digit)},\tTNR: {round(tnr(*stat), digit)}, "
              f"FPR: {round(fpr(*stat), digit)},\tFNR: {round(fnr(*stat), digit)} ")
        print(f"Acc: {round(acc(*stat), digit)}")
        print(f"F1 : {round(f1(*stat), digit)}")
        print(f"Recall: {round(recall(*stat), digit)},\tPrecision: {round(precision(*stat), digit)}")
        print(f"AUROC: {round(auc(curve[0], curve[1]), digit)}")

        print(f"LaTeX:\n{str(round(auc(curve[0], curve[1]), digit))[1:]} "
              f"& {str(round(f1(*stat), digit))[1:]}")

    return implementation


def calc_t5_full_ablation_statistics(ablation_filename):
    def implement(digit=3):
        prediction = T5_full.evaluate_predictions([
            Path("data", "split", "open-web-text", ablation_filename),
            Path("data", "split", "open-gpt-text", ablation_filename),
            Path("data", "split", "open-palm-text", ablation_filename),
            Path("data", "split", "open-llama-text", ablation_filename),
            Path("data", "split", "gpt2-output", ablation_filename)
        ])
        reformulated_predictions = []
        for entry in prediction:
            p_selected = entry["data"][0]
            new_entry = entry.copy()
            new_entry["data"] = np.array([p_selected, 1 - p_selected])
            reformulated_predictions.append(new_entry)

        stat = quick_statistics_binary(reformulated_predictions, "openweb", threshold=0.5)
        curve = get_roc_binary(reformulated_predictions, pos_label="openweb")

        print(f"T5-Sentinel (ours) classifier human-to-generated statistics")
        print(f"Ablation Filename: {ablation_filename}")
        print(f"TPR: {round(tpr(*stat), digit)},\tTNR: {round(tnr(*stat), digit)}, "
              f"FPR: {round(fpr(*stat), digit)},\tFNR: {round(fnr(*stat), digit)} ")
        print(f"Acc: {round(acc(*stat), digit)}")
        print(f"F1 : {round(f1(*stat), digit)}")
        print(f"Recall: {round(recall(*stat), digit)},\tPrecision: {round(precision(*stat), digit)}")
        print(f"AUROC: {round(auc(curve[0], curve[1]), digit)}")

    return implement


def calculate_punctuation_removal(filename: str, pos_label: str, pos_index: int, singleton: str):
    def punctuation_removal(digit = 3):
        prediction = T5_full.evaluate_removals([
            Path("data", "split", "open-web-text", filename),
            Path("data", "split", "open-gpt-text", filename),
            Path("data", "split", "open-palm-text", filename),
            Path("data", "split", "open-llama-text", filename),
            Path("data", "split", "gpt2-output", filename)
        ], singleton)

        reformulated_predictions = []
        for entry in prediction:
            p_selected = entry["data"][pos_index]
            new_entry = entry.copy()
            new_entry["data"] = np.array([p_selected, 1 - p_selected])
            reformulated_predictions.append(new_entry)

        stat = quick_statistics_binary(reformulated_predictions, pos_label, threshold=0.5)
        curve = get_roc_binary(reformulated_predictions, pos_label)

        print(f"Punc remove: {singleton}, label={pos_label} @ {filename}")

        print("- True Positive Rate: {:.03%}".format(tpr(*stat)))
        print("- True Negative Rate: {:.03%}".format(tnr(*stat)))
        print("- False Positive Rate: {:.03%}".format(fpr(*stat)))
        print("- False Negative Rate: {:.03%}".format(fnr(*stat)))
        print("- Accuracy: {:.03%}".format(acc(*stat)))
        print("- F1 Score: {:.03%}".format(f1(*stat)))
        print("- Recall: {:.03%}".format(recall(*stat)))
        print("- Precision: {:.03%}".format(precision(*stat)))
        print("- AUROC: {:.03%}".format(auc(curve[0], curve[1])))

        with open("temp_result.txt", "a") as f:
            f.write(f" & {str(round(auc(curve[0], curve[1]), digit))[1:]} & {str(round(f1(*stat), digit))[1:]}")

    return punctuation_removal


def calculate_punctuation_removal_binary(path: Path, singleton: str):
    def punctuation_removal(digit = 3):
        prediction = T5_full.evaluate_removals([
            Path("data", "split", "open-web-text", "test-dirty.jsonl"),
            path,
        ], singleton)

        reformulated_predictions = []
        for entry in prediction:
            p_selected = entry["data"][0]
            new_entry = entry.copy()
            new_entry["data"] = np.array([p_selected, 1 - p_selected])
            reformulated_predictions.append(new_entry)

        stat = quick_statistics_binary(reformulated_predictions, "openweb", threshold=0.5)
        curve = get_roc_binary(reformulated_predictions, pos_label="openweb")

        print(singleton, path)
        print("- True Positive Rate: {:.03%}".format(tpr(*stat)))
        print("- True Negative Rate: {:.03%}".format(tnr(*stat)))
        print("- False Positive Rate: {:.03%}".format(fpr(*stat)))
        print("- False Negative Rate: {:.03%}".format(fnr(*stat)))
        print("- Accuracy: {:.03%}".format(acc(*stat)))
        print("- F1 Score: {:.03%}".format(f1(*stat)))
        print("- Recall: {:.03%}".format(recall(*stat)))
        print("- Precision: {:.03%}".format(precision(*stat)))
        print("- AUROC: {:.03%}".format(auc(curve[0], curve[1])))

        print(f"LaTeX:\n{str(round(auc(curve[0], curve[1]), digit))[1:]} "
              f"& {str(round(f1(*stat), digit))[1:]}")

    return punctuation_removal



if __name__ == "__main__":
    TASKS = [
        # For table 3. in paper
        calc_t5_full_statistics,
        calc_t5_hidden_statistics,
        calc_openai_baseline_statistics,
        calc_zerogpt_baseline_statistics,

        ################################################################################################################

        # For dataset ablation study in paper
        calc_t5_full_ablation_statistics("test-dirty.jsonl"),
        calc_t5_full_ablation_statistics("test.variant1.jsonl"),
        calc_t5_full_ablation_statistics("test.variant2.jsonl"),
        calc_t5_full_ablation_statistics("test.variant3.jsonl"),
        calc_t5_full_ablation_statistics("test.variant4.jsonl"),

        # For table 4.
        calc_t5_hidden_statistics_detail_full("test-dirty.jsonl", "openweb", 0),
        calc_t5_hidden_statistics_detail_full("test-dirty.jsonl", "chatgpt", 1),
        calc_t5_hidden_statistics_detail_full("test-dirty.jsonl", "palm"   , 2),
        calc_t5_hidden_statistics_detail_full("test-dirty.jsonl", "llama"  , 3),
        calc_t5_hidden_statistics_detail_full("test-dirty.jsonl", "gpt2_xl", 4),

        ################################################################################################################

        # For table 5.
        calc_openai_baseline_statistics_detail(Path("data", "baselines", "openai_classifier_output", "open-gpt-text.jsonl")),
        calc_openai_baseline_statistics_detail(Path("data", "baselines", "openai_classifier_output", "open-palm-text.jsonl")),
        calc_openai_baseline_statistics_detail(Path("data", "baselines", "openai_classifier_output", "open-llama-text.jsonl")),
        calc_openai_baseline_statistics_detail(Path("data", "baselines", "openai_classifier_output", "gpt2-output.jsonl")),

        calc_zerogpt_baseline_statistics_detail(Path("data", "baselines", "zerogpt_classifier_output", "open-gpt-text.jsonl")),
        calc_zerogpt_baseline_statistics_detail(Path("data", "baselines", "zerogpt_classifier_output", "open-palm-text.jsonl")),
        calc_zerogpt_baseline_statistics_detail(Path("data", "baselines", "zerogpt_classifier_output", "open-llama-text.jsonl")),
        calc_zerogpt_baseline_statistics_detail(Path("data", "baselines", "zerogpt_classifier_output", "gpt2-output.jsonl")),

        calc_t5_full_statistics_detail("test-dirty.jsonl", "chatgpt", 1),
        calc_t5_full_statistics_detail("test-dirty.jsonl", "palm"   , 2),
        calc_t5_full_statistics_detail("test-dirty.jsonl", "llama"  , 3),
        calc_t5_full_statistics_detail("test-dirty.jsonl", "gpt2_xl", 4),

        calc_t5_hidden_statistics_detail(Path("data", "split", "open-gpt-text", "test-dirty.jsonl")),
        calc_t5_hidden_statistics_detail(Path("data", "split", "open-palm-text", "test-dirty.jsonl")),
        calc_t5_hidden_statistics_detail(Path("data", "split", "open-llama-text", "test-dirty.jsonl")),
        calc_t5_hidden_statistics_detail(Path("data", "split", "gpt2-output", "test-dirty.jsonl")),

        ################################################################################################################

        # For table 6.
        calc_t5_full_statistics_detail("test-dirty.jsonl", "openweb", 0),
        calc_t5_full_statistics_detail("test-dirty.jsonl", "chatgpt", 1),
        calc_t5_full_statistics_detail("test-dirty.jsonl", "palm"   , 2),
        calc_t5_full_statistics_detail("test-dirty.jsonl", "llama"  , 3),
        calc_t5_full_statistics_detail("test-dirty.jsonl", "gpt2_xl", 4),

        calc_t5_full_statistics_detail("test.variant1.jsonl", "openweb", 0),
        calc_t5_full_statistics_detail("test.variant1.jsonl", "chatgpt", 1),
        calc_t5_full_statistics_detail("test.variant1.jsonl", "palm", 2),
        calc_t5_full_statistics_detail("test.variant1.jsonl", "llama", 3),
        calc_t5_full_statistics_detail("test.variant1.jsonl", "gpt2_xl", 4),

        calc_t5_full_statistics_detail("test.variant2.jsonl", "openweb", 0),
        calc_t5_full_statistics_detail("test.variant2.jsonl", "chatgpt", 1),
        calc_t5_full_statistics_detail("test.variant2.jsonl", "palm", 2),
        calc_t5_full_statistics_detail("test.variant2.jsonl", "llama", 3),
        calc_t5_full_statistics_detail("test.variant2.jsonl", "gpt2_xl", 4),

        calc_t5_full_statistics_detail("test.variant3.jsonl", "openweb", 0),
        calc_t5_full_statistics_detail("test.variant3.jsonl", "chatgpt", 1),
        calc_t5_full_statistics_detail("test.variant3.jsonl", "palm", 2),
        calc_t5_full_statistics_detail("test.variant3.jsonl", "llama", 3),
        calc_t5_full_statistics_detail("test.variant3.jsonl", "gpt2_xl", 4),

        calculate_punctuation_removal("test-dirty.jsonl", "openweb", 0, "."),
        calculate_punctuation_removal("test-dirty.jsonl", "chatgpt", 1, "."),
        calculate_punctuation_removal("test-dirty.jsonl", "palm", 2, "."),
        calculate_punctuation_removal("test-dirty.jsonl", "llama", 3, "."),
        calculate_punctuation_removal("test-dirty.jsonl", "gpt2_xl", 4, "."),

        calculate_punctuation_removal("test-dirty.jsonl", "openweb", 0, ","),
        calculate_punctuation_removal("test-dirty.jsonl", "chatgpt", 1, ","),
        calculate_punctuation_removal("test-dirty.jsonl", "palm", 2, ","),
        calculate_punctuation_removal("test-dirty.jsonl", "llama", 3, ","),
        calculate_punctuation_removal("test-dirty.jsonl", "gpt2_xl", 4, ","),

        calculate_punctuation_removal("test-dirty.jsonl", "openweb", 0, "?"),
        calculate_punctuation_removal("test-dirty.jsonl", "chatgpt", 1, "?"),
        calculate_punctuation_removal("test-dirty.jsonl", "palm", 2, "?"),
        calculate_punctuation_removal("test-dirty.jsonl", "llama", 3, "?"),
        calculate_punctuation_removal("test-dirty.jsonl", "gpt2_xl", 4, "?"),

        calculate_punctuation_removal("test-dirty.jsonl", "openweb", 0, "!"),
        calculate_punctuation_removal("test-dirty.jsonl", "chatgpt", 1, "!"),
        calculate_punctuation_removal("test-dirty.jsonl", "palm", 2, "!"),
        calculate_punctuation_removal("test-dirty.jsonl", "llama", 3, "!"),
        calculate_punctuation_removal("test-dirty.jsonl", "gpt2_xl", 4, "!"),

        calculate_punctuation_removal("test-dirty.jsonl", "openweb", 0, ":"),
        calculate_punctuation_removal("test-dirty.jsonl", "chatgpt", 1, ":"),
        calculate_punctuation_removal("test-dirty.jsonl", "palm", 2, ":"),
        calculate_punctuation_removal("test-dirty.jsonl", "llama", 3, ":"),
        calculate_punctuation_removal("test-dirty.jsonl", "gpt2_xl", 4, ":"),

        calculate_punctuation_removal("test-dirty.jsonl", "openweb", 0, "'"),
        calculate_punctuation_removal("test-dirty.jsonl", "chatgpt", 1, "'"),
        calculate_punctuation_removal("test-dirty.jsonl", "palm", 2, "'"),
        calculate_punctuation_removal("test-dirty.jsonl", "llama", 3, "'"),
        calculate_punctuation_removal("test-dirty.jsonl", "gpt2_xl", 4, "'"),

        calculate_punctuation_removal("test-dirty.jsonl", "openweb", 0, "\""),
        calculate_punctuation_removal("test-dirty.jsonl", "chatgpt", 1, "\""),
        calculate_punctuation_removal("test-dirty.jsonl", "palm", 2, "\""),
        calculate_punctuation_removal("test-dirty.jsonl", "llama", 3, "\""),
        calculate_punctuation_removal("test-dirty.jsonl", "gpt2_xl", 4, "\""),

        calculate_punctuation_removal("test-dirty.jsonl", "openweb", 0, "*"),
        calculate_punctuation_removal("test-dirty.jsonl", "chatgpt", 1, "*"),
        calculate_punctuation_removal("test-dirty.jsonl", "palm", 2, "*"),
        calculate_punctuation_removal("test-dirty.jsonl", "llama", 3, "*"),
        calculate_punctuation_removal("test-dirty.jsonl", "gpt2_xl", 4, "*"),

        calc_t5_full_statistics_detail("test.variant4.jsonl", "openweb", 0),
        calc_t5_full_statistics_detail("test.variant4.jsonl", "chatgpt", 1),
        calc_t5_full_statistics_detail("test.variant4.jsonl", "palm", 2),
        calc_t5_full_statistics_detail("test.variant4.jsonl", "llama", 3),
        calc_t5_full_statistics_detail("test.variant4.jsonl", "gpt2_xl", 4),
    ]

    for task in TASKS:
        print("Executing task: ", task.__name__)
        task()
        print("\n")
