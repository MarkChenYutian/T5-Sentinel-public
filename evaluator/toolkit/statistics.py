from typing import Sequence

import numpy as np

from pipeline import P


def quick_statistics_binary(prediction: Sequence[P.ArrayEntry], pos_label, threshold=0.5):
    """
    :param prediction: Binary prediction in form of [P(positive), P(negative)]
    :param pos_label: Label name (entry["extra"]["source"]) for positive entry
    :param threshold: Threshold probability for being classified of ChatGPT
    :return: TP, TN, FP, FN
    """
    TP, TN, FP, FN = 0, 0, 0, 0
    for entry in prediction:
        assert entry["data"].size == 2, "Non-binary data input received"

        p_positive, p_web = entry["data"][0], entry["data"][1]
        pred_positive = p_positive >= threshold

        label_positive = entry["extra"]["source"] == pos_label
        if pred_positive and label_positive: TP += 1
        elif pred_positive and (not label_positive): FP += 1
        elif (not pred_positive) and label_positive: FN += 1
        else: TN += 1
    return TP, TN, FP, FN


def report_statistics(TP, TN, FP, FN):
    TPR = tpr(TP, TN, FP, FN)
    TNR = tnr(TP, TN, FP, FN)
    FPR = fpr(TP, TN, FP, FN)
    FNR = fnr(TP, TN, FP, FN)
    print(f"True Positive: {TP} \t| True Negative: {TN}")
    print(f"False Positive:{FP} \t| False Negative:{FN}")
    print(f"True Positive Rate:  {round(TPR * 100, 2)}\%")
    print(f"True Negative Rate:  {round(TNR * 100, 2)}\%")
    print(f"False Positive Rate: {round(FPR * 100, 2)}\%")
    print(f"False Negative Rate: {round(FNR * 100, 2)}\%")
    print(f"Accuracy: {round(((TP + TN) / (TP + TN + FP + FN)) * 100, 2)}\%")
    print(f"F1 Score: {round((TP) / (TP + 0.5 * (FP + FN)), 2)}")

    print("LaTeX Usable-version\n")

    print(
    f"{round(((TP + TN) / (TP + TN + FP + FN)) * 100, 2)}\%", "&"
    f"{round(TPR * 100, 2)}\%, ({TP})", "&",
    f"{round(TNR * 100, 2)}\%, ({TN})", "&",
    f"{round(FPR * 100, 2)}\%, ({FP})", "&",
    f"{round(FNR * 100, 2)}\%, ({FN})", "\\\\"
    )

def tpr(TP, TN, FP, FN): return TP / (TP + FN)
def tnr(TP, TN, FP, FN): return TN / (TN + FP)
def fpr(TP, TN, FP, FN): return FP / (FP + TN)
def fnr(TP, TN, FP, FN): return FN / (FN + TP)

recall = tpr

def precision(TP, TN, FP, FN): return TP / (TP + FP)

def acc(TP, TN, FP, FN): return (TP + TN) / (TP + TN + FP + FN)

def f1(TP, TN, FP, FN): return TP / (TP + 0.5 * (FP + FN))

def calculate_confusion_matrix(predictions: Sequence[P.ArrayEntry], classes: Sequence[str]) -> np.ndarray:
    k = len(classes)
    confusion = np.zeros((k, k))

    for entry in predictions:
        label = entry["extra"]["source"]
        label_idx = classes.index(label)
        pred_idx = np.argmax(entry["data"])
        confusion[label_idx, pred_idx] += 1

    return confusion
