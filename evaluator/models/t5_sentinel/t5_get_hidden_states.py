import sys

sys.path.append(".")

import typing as T
from pipeline import PipelineExecutor, P
from evaluator.models.t5_sentinel.t5_pipeline import (
    ExecuteT5,
    T5PredictToLogits,
    T5PredictToHidden,
)
from pathlib import Path
from memoizer import memoize

CHECKPOINT = "T5Sentinel.0613.pt"


def list_reduce(l1: T.List, l2: T.List) -> T.List:
    l1.extend(l2)
    return l1


def argeq(arg1, arg2):
    return arg1[0] == arg2[0]


def binary_eq(arg1, arg2):
    return arg1[0] == arg2[0] and arg1[1] == arg2[1]


@memoize(cache_path=Path("./cache/t5_full_predicts.pt"), arg_eq=argeq)
def evaluate_prediction_impl(from_file: Path) -> T.Sequence[P.ArrayEntry]:
    pipe = (
        P.FromJsonStr()
        >> ExecuteT5(Path("./data/checkpoint", CHECKPOINT))
        >> T5PredictToLogits()
        >> P.ToSingletonList(T.Optional[P.ArrayEntry])
    )

    executor = PipelineExecutor(worker_num=1)
    predicts = executor.sequential_mapreduce(
        map_fn=pipe,
        from_files=[from_file],
        identity=list(),
        reduce_fn=list_reduce,
        verbose=True,
    )
    return predicts


@memoize(cache_path=Path("./cache/t5_full_hiddens.pt"), arg_eq=argeq)
def evaluate_hiddens_impl(from_file: Path) -> T.Sequence[P.ArrayEntry]:
    pipe = (
        P.FromJsonStr()
        >> ExecuteT5(Path("./data/checkpoint", CHECKPOINT))
        >> T5PredictToHidden()
        >> P.ToSingletonList(T.Optional[P.ArrayEntry])
    )

    executor = PipelineExecutor(worker_num=1)
    hiddens = executor.sequential_mapreduce(
        map_fn=pipe,
        from_files=[from_file],
        identity=list(),
        reduce_fn=list_reduce,
        verbose=True,
    )
    return hiddens


@memoize(cache_path=Path("./cache/t5_punctuation_removal.pt"), arg_eq=binary_eq)
def evaluate_removals_impl(from_file: Path, singleton: str) -> T.Sequence[P.ArrayEntry]:
    pipe = (
        P.FromJsonStr()
        >> P.RemoveSingleton(singleton)
        >> ExecuteT5(Path("./data/checkpoint", CHECKPOINT))
        >> T5PredictToLogits()
        >> P.ToSingletonList(T.Optional[P.ArrayEntry])
    )

    executor = PipelineExecutor(worker_num=1)
    predicts = executor.sequential_mapreduce(
        map_fn=pipe,
        from_files=[from_file],
        identity=list(),
        reduce_fn=list_reduce,
        verbose=True,
    )
    return predicts


def evaluate_predictions(from_files: T.Sequence[Path]) -> T.Sequence[P.ArrayEntry]:
    result = []
    for from_file in from_files:
        result.extend(evaluate_prediction_impl(from_file))
    return result


def evaluate_hidden_states(from_files: T.Sequence[Path]) -> T.Sequence[P.ArrayEntry]:
    result = []
    for from_file in from_files:
        result.extend(evaluate_hiddens_impl(from_file))
    return result


def evaluate_removals(
    from_files: T.Sequence[Path], singleton: str
) -> T.Sequence[P.ArrayEntry]:
    result = []
    for from_file in from_files:
        result.extend(evaluate_removals_impl(from_file, singleton))
    return result
