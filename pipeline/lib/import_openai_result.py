from pipeline import P, PipelineExecutor
from pathlib import Path
from memoizer import memoize
import numpy as np
import typing as Tp
import json
import math


class LoadOpenAIPredictionResult(P.Pipeline[Tp.Optional[str], Tp.Optional[P.ArrayEntry]]):
    def __call__(self, x: Tp.Optional[str]) -> Tp.Optional[P.ArrayEntry]:
        if x is None: return None

        entry = json.loads(x)
        uid, extra = entry["uid"], entry["extra"]
        pred_result = entry["res"]
        if len(pred_result["choices"][0]["logprobs"]["top_logprobs"]) == 0:
            prob_human = 0.
        elif "!" in pred_result["choices"][0]["logprobs"]["top_logprobs"][0]:
            prob_human = math.exp(pred_result["choices"][0]["logprobs"]["top_logprobs"][0]["!"])
        else:
            prob_human = 1e-5
        prob_vec = np.array([prob_human, 1 - prob_human])
        return {"uid": uid, "extra": extra, "data": prob_vec}


def list_reduce(l1: Tp.List, l2: Tp.List) -> Tp.List:
    l1.extend(l2)
    return l1


def argeq(a, b): return a[0] == b[0]


@memoize(Path("cache", "openai_prediction.pt"), arg_eq=argeq)
def import_openai_prediction_result_impl(path: Path):
    executor = PipelineExecutor(worker_num=4)
    predictions = executor.sequential_mapreduce(
        LoadOpenAIPredictionResult() >> P.ToSingletonList(input_type=Tp.Optional[P.ArrayEntry]),
        from_files=[path],
        identity=[],
        reduce_fn=list_reduce,
        verbose=True
    )
    return predictions

def import_openai_prediction_result():
    openai_gpt   = import_openai_prediction_result_impl(
        Path("data", "baselines", "openai_classifier_output", "open-gpt-text.jsonl")
    )
    openai_llama = import_openai_prediction_result_impl(
        Path("data", "baselines", "openai_classifier_output", "open-llama-text.jsonl")
    )
    openai_palm  = import_openai_prediction_result_impl(
        Path("data", "baselines", "openai_classifier_output", "open-palm-text.jsonl")
    )
    openai_web   = import_openai_prediction_result_impl(
        Path("data", "baselines", "openai_classifier_output", "open-web-text.jsonl")
    )
    openai_gpt2  = import_openai_prediction_result_impl(
        Path("data", "baselines", "openai_classifier_output", "gpt2-output.jsonl")
    )
    result = openai_gpt + openai_llama + openai_palm + openai_web + openai_gpt2
    return result


if __name__ == "__main__":
    import_openai_prediction_result()
