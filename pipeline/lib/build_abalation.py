from pathlib import Path
from pipeline import P
from tqdm import tqdm


def build_clean_variants(root_path: Path):
    clean_1 = P.StripNewline() >> P.WriteExtra({"variant": "clean-x-newline"}) >> P.ToJsonStr() >> P.WriteTo(Path(root_path, "test.variant1.jsonl"), write_mode="a")
    clean_2 = P.CastUnicode()  >> P.WriteExtra({"variant": "clean-x-unicode"}) >> P.ToJsonStr() >> P.WriteTo(Path(root_path, "test.variant2.jsonl"), write_mode="a")
    clean_3 = P.RemovePunc()   >> P.WriteExtra({"variant": "clean-x-punct"})   >> P.ToJsonStr() >> P.WriteTo(Path(root_path, "test.variant3.jsonl"), write_mode="a")
    clean_4 = P.ToLower()      >> P.WriteExtra({"variant": "clean-x-lower"})   >> P.ToJsonStr() >> P.WriteTo(Path(root_path, "test.variant4.jsonl"), write_mode="a")

    process_pipeline = P.FromJsonStr() >> P.Tee(clean_1) >> P.Tee(clean_2) >> P.Tee(clean_3) >> clean_4

    with open(Path(root_path, "test-dirty.jsonl"), "r") as f:
        lines = f.read().strip().splitlines()

    for line in tqdm(lines, desc=str(root_path)): process_pipeline(line)


if __name__ == "__main__":
    build_clean_variants(Path("data", "split", "open-palm-text"))
    build_clean_variants(Path("data", "split", "open-web-text"))
    build_clean_variants(Path("data", "split", "open-gpt-text"))
    build_clean_variants(Path("data", "split", "gpt2-output"))
    build_clean_variants(Path("data", "split", "open-llama-text"))
