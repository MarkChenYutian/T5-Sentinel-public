import torch
import typing as Tp
from pathlib import Path
from pipeline import P

from detector.t5_hidden.model import Sentinel
from detector.t5_hidden.types import SentinelOutput
from transformers import T5TokenizerFast as Tokenizer


class T5PredictionOutput(Tp.TypedDict):
    hidden: Tp.List[torch.Tensor]
    predict: torch.Tensor
    input: P.TextEntry


class ExecuteT5Hidden(
    P.Pipeline[Tp.Optional[P.TextEntry], Tp.Optional[T5PredictionOutput]]
):
    def __init__(self, weight_path: Path, backbone_name: str = "t5-small"):
        assert weight_path.exists()
        super().__init__()

        checkpoint = torch.load(weight_path)
        model = Sentinel()
        model.load_state_dict(checkpoint["model"])

        self.model = model.to("cuda")
        self.model.config.mode = "interpret"
        self.model.eval()
        self.tokenizer = Tokenizer.from_pretrained(backbone_name, model_max_length=512)

    def __call__(
        self, entry: Tp.Optional[P.TextEntry]
    ) -> Tp.Optional[T5PredictionOutput]:
        if entry is None:
            return None
        text_tokenized = self.tokenizer.batch_encode_plus(
            (entry["text"],), padding=True, truncation=True, return_tensors="pt"
        )
        result: SentinelOutput = self.model.forward(
            text_tokenized.input_ids.cuda(),
            text_tokenized.attention_mask.cuda(),
        )

        logits = result.probabilities.detach().cpu()
        hiddens = [
            h.detach().cpu() for h in result.huggingface.decoder_hidden_states[0]
        ]

        return {"hidden": hiddens, "predict": logits, "input": entry}


class T5HiddenPredictToLogits(
    P.Pipeline[Tp.Optional[T5PredictionOutput], Tp.Optional[P.ArrayEntry]]
):
    def __call__(
        self, prediction: Tp.Optional[T5PredictionOutput]
    ) -> Tp.Optional[P.ArrayEntry]:
        if prediction is None:
            return None

        uid: str = (
            prediction["input"]["uid"]
            if prediction["input"]["uid"] is not None
            else "No-UID-placeholder"
        )
        return {
            "uid": uid,
            "data": prediction["predict"].numpy(),
            "extra": prediction["input"]["extra"],
        }


class T5PredictToHidden(
    P.Pipeline[Tp.Optional[T5PredictionOutput], Tp.Optional[P.ArrayEntry]]
):
    def __call__(
        self, prediction: Tp.Optional[T5PredictionOutput]
    ) -> Tp.Optional[P.ArrayEntry]:
        if prediction is None:
            return None

        uid: str = (
            prediction["input"]["uid"]
            if prediction["input"]["uid"] is not None
            else "No-UID-placeholder"
        )
        return {
            "uid": uid,
            "data": prediction["hidden"],
            "extra": prediction["input"]["extra"],
        }


if __name__ == "__main__":
    pipe = (
        ExecuteT5Hidden(Path("./data/checkpoint/t5-small.0621.hidden.pt"))
        >> T5HiddenPredictToLogits()
    )
    result = pipe(
        {
            "uid": "001",
            "text": """What is this?\n\nThis is a neat little plugin to give more options to the webp daemon. It's a vector graphic library when used together with a webscape window, and although smoothies aren't a thing yet. It uses dot-comparison for vector shapes, and is even able to do line spaces. The is a fork of dvipng (although, it won't require you to compile the files during the last step, just download the latest source code) which is already being developed and tested byDendi Samantasan . It's also opened source, so anyone else can add back and forth support to the program. This is the first preview of this file, so it will likely change over time. There's a mailing list for discussion and bug reporting at""",
            "extra": None,
        }
    )
    print(result)
