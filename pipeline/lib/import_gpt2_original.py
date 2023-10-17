import json
import typing as T
from pipeline import P, PipelineExecutor
from pathlib import Path


class rawGPT2Loader(P.Pipeline[str, T.Optional[P.TextEntry]]):
    def __call__(self, x: str) -> P.TextEntry:
        entry = json.loads(x)
        result = {"uid": f"gpt2_{entry['id']}", "text": entry["text"], "extra": {"variant": "original", "source": "gpt2_xl"}}
        return result


def reduce_list(l1: T.List[P.TextEntry], l2: T.List[P.TextEntry]):
    l1.extend(l2)
    return l1


def sample_gpt2(subset_names):
    sample_pipeline = rawGPT2Loader() \
                      >> P.RandomFilter(block_factor=0.75) \
                      >> P.ToJsonStr() \
                      >> P.ToSingletonList(input_type=T.Optional[str])

    executor = PipelineExecutor(worker_num=3)
    sampled = executor.parallel_mapreduce(
        sample_pipeline,
        from_files=[Path("./data/original/gpt2-output", subset) for subset in subset_names],
        identity=[],
        reduce_fn=reduce_list,
        verbose=True
    )

    print(f"Sampled {len(sampled)} lines")
    with open(Path("data", "original", "gpt2-output", "sampled_gpt2.jsonl"), "w") as f:
        for line in sampled: f.write(line + "\n")


if __name__ == "__main__":
    subsets = ["xl-1542M.test.jsonl", "xl-1542M.valid.jsonl", "xl-1542M.train.jsonl"]
    sample_gpt2(subsets)

