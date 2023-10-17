import typing as Tp
import json
import re
import string
import unidecode
from pathlib import Path

from ..pipeline_base import Pipeline

I = Tp.TypeVar("I")
O = Tp.TypeVar("O")
WHITELIST = string.whitespace + string.digits + string.ascii_letters


class TextEntry(Tp.TypedDict):
    text: str
    uid: Tp.Optional[str]
    extra: Tp.Optional[dict]


class FromJsonStr(Pipeline[Tp.Optional[str], Tp.Optional[TextEntry]]):
    def __call__(self, x):
        if x is None: return None
        obj = json.loads(x, strict=False)
        if "uid" not in obj: obj["uid"] = None
        if "extra" not in obj: obj["extra"] = None
        return obj


class ToJsonStr(Pipeline[Tp.Optional[TextEntry], Tp.Optional[str]]):
    def __call__(self, x):
        if x is None: return None
        return json.dumps(x)


class ToStr(Pipeline[Tp.Optional[TextEntry], Tp.Optional[str]]):
    def __call__(self, x):
        if x is None: return None
        return x["text"]


class ToUID(Pipeline[Tp.Optional[TextEntry], Tp.Optional[str]]):
    def __call__(self, obj):
        if obj is None: return None
        return obj["uid"]


class StripNewline(Pipeline[Tp.Optional[TextEntry], Tp.Optional[TextEntry]]):
    def __call__(self, obj):
        if obj is None: return obj
        obj["text"] = re.sub(r"\n+", r"\n", obj["text"])
        return obj


class CastUnicode(Pipeline[Tp.Optional[TextEntry], Tp.Optional[TextEntry]]):
    def __call__(self, obj):
        if obj is None: return obj
        obj["text"] = unidecode.unidecode(obj["text"])
        return obj


class WriteExtra(Pipeline[Tp.Optional[TextEntry], Tp.Optional[TextEntry]]):
    def __init__(self, extra_fields: dict):
        super().__init__()
        self.extra_fields = extra_fields

    def __call__(self, obj):
        if obj is None: return obj
        if obj["extra"] is None: obj["extra"] = dict()

        for key in self.extra_fields:
            obj["extra"][key] = self.extra_fields[key]
        return obj


class ToLower(Pipeline[Tp.Optional[TextEntry], Tp.Optional[TextEntry]]):
    def __call__(self, obj):
        if obj is None: return obj
        obj["text"] = obj["text"].lower()
        return obj


class RemovePunc(Pipeline[Tp.Optional[TextEntry], Tp.Optional[TextEntry]]):
    def __call__(self, obj):
        if obj is None: return obj
        obj["text"] = "".join(filter(WHITELIST.__contains__, obj["text"]))
        return obj


class RemoveSingleton(Pipeline[Tp.Optional[TextEntry], Tp.Optional[TextEntry]]):
    def __init__(self, singleToRemove: str):
        super().__init__()
        self.singleToRemove = singleToRemove

    def __call__(self, obj):
        if obj is None: return obj
        obj["text"] = obj["text"].replace(self.singleToRemove, "")
        return obj

class RemoveContSpace(Pipeline[Tp.Optional[TextEntry], Tp.Optional[TextEntry]]):
    def __call__(self, obj: Tp.Optional[TextEntry]):
        if obj is None: return obj
        obj["text"] = re.sub(r'\s+', ' ', obj["text"])
        obj["text"] = obj["text"].strip()
        return obj


class FilterTextEntry(Pipeline[Tp.Optional[TextEntry], Tp.Optional[TextEntry]]):
    def __init__(self, rule: Tp.Callable[[TextEntry], bool]):
        super().__init__()
        self.filter_rule = rule

    def __call__(self, obj):
        return obj if (obj is not None) and (not self.filter_rule(obj)) else None


class FilterIf_UID_NotInFile(Pipeline[Tp.Optional[TextEntry], Tp.Optional[TextEntry]]):
    def __init__(self, file_path: Path):
        super().__init__()
        assert (file_path.exists())
        loader_pipeline = FromJsonStr() >> ToUID()

        self.uid_whitelist = set()
        with open(file_path, "r") as f:
            for line in f.read().strip().split("\n"):
                if line != "": self.uid_whitelist.add(loader_pipeline(line))

    def __call__(self, obj: Tp.Optional[TextEntry]):
        if obj is None or obj["uid"] not in self.uid_whitelist: return None
        return obj


class NegateFilter(Pipeline[Tp.Optional[TextEntry], Tp.Optional[TextEntry]]):
    def __init__(self, filter_pipe: Pipeline):
        super().__init__()
        self.filter = filter_pipe

    def __call__(self, obj: Tp.Optional[TextEntry]):
        if obj is None: return None
        result = self.filter(obj)
        return obj if result is None else None
