import yaml
import typing as Tp

from pipeline import P, PipelineExecutor
from pathlib import Path
from vertexai.preview.language_models import TextGenerationModel


class PaLMRequest(P.Pipeline[Tp.Optional[P.TextEntry], Tp.Optional[P.TextEntry]]):
    def __init__(self, **config):
        super().__init__()
        print(f"Initializing {repr(self)} Pipeline Component with configuration:")
        print(config)

        self.config= config
        self.retry = config["retry"]
        self.model = TextGenerationModel.from_pretrained(self.config["ModelName"])

    def __call__(self, entry: Tp.Optional[P.TextEntry]) -> Tp.Optional[P.TextEntry]:
        if entry is None: return None
        try:
            rephrased = ""
            for _ in range(self.retry):
                rephrased = self.model.predict(
                    f"""Rephrase the following paragraph by paragraph:\n "{entry["text"]}" """,
                    temperature=self.config["Temperature"],
                    max_output_tokens=self.config["MaxDecodeSteps"],
                    top_p=self.config["top_p"],
                    top_k=self.config["top_k"]
                )
                if rephrased.text != "": break
            if rephrased.text == "": return None
            result = {"uid": entry["uid"], "text": rephrased.text, "extra": entry["extra"]}
            return result
        except Exception as e:
            print(f"[x] - \t Exception caught: {entry['uid']} - {e}")
            return None


if __name__ == "__main__":
    CFG_PATH = Path("./generator/palm/palm_client.yaml")
    subsets = ["urlsf_subset03.jsonl", "urlsf_subset00.jsonl"]

    with open(CFG_PATH, "r") as f: global_cfg = yaml.safe_load(f)
    # Print UID if something is successfully rephrased by PaLM
    print_uid_side_effect = P.ToUID() >> P.Print(prefix="[√] - \t", omit_none=True)
    duplicate_filters = [
        P.FilterIf_UID_NotInFile(Path("data", "original", "open-palm-text", subset)) for subset in subsets
    ]
    dedupe_filter = duplicate_filters[0]
    for i in range(1, len(duplicate_filters)): dedupe_filter = dedupe_filter >> duplicate_filters[i]

    pipeline = P.FromJsonStr() \
               >> P.NegateFilter(dedupe_filter) \
               >> P.RandomFilter(block_factor=0.98) \
               >> P.WriteExtra({"source": "palm", "variant": "original"}) \
               >> P.RateControl(**global_cfg["RateControl"]) \
               >> PaLMRequest(**global_cfg["Config"]) \
               >> P.Tee(print_uid_side_effect) \
               >> P.ToJsonStr()


    executor = PipelineExecutor(worker_num=1)
    executor.sequantial_file_mapping(
        pipeline,
        from_files=[Path("./data/original/open-web-text", subset) for subset in subsets],
        to_files=[Path("./data/original/open-palm-text", subset) for subset in subsets],
        write_mode="a"
    )
