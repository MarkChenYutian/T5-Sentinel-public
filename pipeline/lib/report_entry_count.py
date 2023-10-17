from pathlib import Path


def report(source_path: str, file_name: str):
    with open(Path("data", "split", source_path, file_name), "r") as f:
        lines = f.read().strip().split("\n")
    print(f"From: {source_path}, \tSubset: {file_name}, \tCount: {len(lines)}")


if __name__ == "__main__":
    sources = ["open-web-text", "open-gpt-text", "open-palm-text", "open-llama-text", "gpt2-output"]
    subsets = ["train-dirty.jsonl", "valid-dirty.jsonl", "test-dirty.jsonl"]

    for source in sources:
        for subset in subsets:
            report(source, subset)
