from pipeline import P, PipelineExecutor, utils
from pathlib import Path
import typing as Tp


Human_Data = [
    Path("data", "split", "open-web-text", "train-dirty.jsonl"),
    Path("data", "split", "open-web-text", "test-dirty.jsonl"),
    Path("data", "split", "open-web-text", "valid-dirty.jsonl")
]

GPT3_Data = [
    Path("data", "split", "open-gpt-text", "test-dirty.jsonl"),
    Path("data", "split", "open-gpt-text", "train-dirty.jsonl"),
    Path("data", "split", "open-gpt-text", "valid-dirty.jsonl")
]

GPT2_Data = [
    Path("data", "split", "gpt2-output", "train-dirty.jsonl"),
    Path("data", "split", "gpt2-output", "valid-dirty.jsonl"),
    Path("data", "split", "gpt2-output", "test-dirty.jsonl")
]

PaLM_Data = [
    Path("data", "split", "open-palm-text", "train-dirty.jsonl"),
    Path("data", "split", "open-palm-text", "valid-dirty.jsonl"),
    Path("data", "split", "open-palm-text", "test-dirty.jsonl"),
]

LLaMA_Data = [
    Path("data", "split", "open-llama-text", "train-dirty.jsonl"),
    Path("data", "split", "open-llama-text", "valid-dirty.jsonl"),
    Path("data", "split", "open-llama-text", "test-dirty.jsonl"),
]

def load_data(files: Tp.List[Path]) -> Tp.Sequence[str]:
    executor = PipelineExecutor(worker_num=min(len(files), 8))
    result = executor.sequential_mapreduce(
        map_fn=P.FromJsonStr() >> P.ToStr() >> P.ToSingletonList(input_type=Tp.Optional[str]),
        from_files=files,
        identity=[],
        reduce_fn=utils.reduce_list
    )
    return result
