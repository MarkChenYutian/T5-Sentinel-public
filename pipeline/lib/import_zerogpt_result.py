from pipeline import P, PipelineExecutor
from pathlib import Path
from memoizer import memoize
import numpy as np
import typing as Tp
import json


class LoadZeroGPTPrediction(P.Pipeline[Tp.Optional[str], Tp.Optional[P.ArrayEntry]]):
    def __call__(self, x: Tp.Optional[str]) -> Tp.Optional[P.ArrayEntry]:
        if x is None: return None

        entry = json.loads(x)
        uid, extra = entry["uid"], entry["extra"]
        res = entry["res"]
        if not res["success"]: return None

        prob_human = float(res["data"]["isHuman"]) / 100.
        prob_vec = np.array([prob_human, 1. - prob_human])
        return {"uid": uid, "extra": extra, "data": prob_vec}


def list_reduce(l1: Tp.List, l2: Tp.List) -> Tp.List:
    l1.extend(l2)
    return l1


def argeq(a, b): return a[0] == b[0]


@memoize(Path("cache", "zerogpt_prediction.pt"), arg_eq=argeq)
def import_zerogpt_prediction_result_impl(path: Path):
    executor = PipelineExecutor(worker_num=4)
    predictions = executor.sequential_mapreduce(
        LoadZeroGPTPrediction() >> P.ToSingletonList(input_type=Tp.Optional[P.ArrayEntry]),
        from_files=[path],
        identity=[],
        reduce_fn=list_reduce,
        verbose=True
    )
    return predictions

def import_zerogpt_prediction_result():
    openai_gpt   = import_zerogpt_prediction_result_impl(
        Path("data", "baselines", "zerogpt_classifier_output", "open-gpt-text.jsonl")
    )
    openai_llama = import_zerogpt_prediction_result_impl(
        Path("data", "baselines", "zerogpt_classifier_output", "open-llama-text.jsonl")
    )
    openai_palm  = import_zerogpt_prediction_result_impl(
        Path("data", "baselines", "zerogpt_classifier_output", "open-palm-text.jsonl")
    )
    openai_web   = import_zerogpt_prediction_result_impl(
        Path("data", "baselines", "zerogpt_classifier_output", "open-web-text.jsonl")
    )
    openai_gpt2  = import_zerogpt_prediction_result_impl(
        Path("data", "baselines", "zerogpt_classifier_output", "gpt2-output.jsonl")
    )
    result = openai_gpt + openai_llama + openai_palm + openai_web + openai_gpt2
    return result


if __name__ == "__main__":
    import_zerogpt_prediction_result()
