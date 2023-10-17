import random
import copy
import typing as Tp
from ..pipeline_base import Pipeline

from pathlib import Path
import hashlib

I = Tp.TypeVar("I")
O = Tp.TypeVar("O")


class GetFileMD5(Pipeline[Path, Tp.Optional[str]]):
    def __init__(self, must_exist=False):
        super().__init__()
        self.must_exist = False

    def __call__(self, path: Path) -> Tp.Optional[str]:
        if not path.exists():
            if self.must_exist:
                raise FileNotFoundError(f"Try to compute MD5 of {path}, but file not exist.")
            else:
                return None

        md5 = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""): md5.update(chunk)
        return md5.hexdigest()


class Tee(Pipeline[I, I]):
    def __init__(self, side_effect_pipe):
        super().__init__()
        self.pipe = side_effect_pipe

    def __call__(self, obj):
        self.pipe(copy.deepcopy(obj))
        return obj


class Apply(Pipeline[I, O]):
    def __init__(self, input_type: Tp.Type[I], output_type: Tp.Type[O], function: Tp.Callable[[I], O]):
        super().__init__()
        self.IN_TYPE = input_type
        self.OUT_TYPE = output_type
        self.function = function

    def __call__(self, obj):
        return self.function(obj)


class RandomFilter(Pipeline[I, I]):
    def __init__(self, block_factor: float):
        super().__init__()
        assert (0 <= block_factor <= 1)

        self.block_factor = block_factor

    def __call__(self, x: I):
        if x is None: return None
        if random.random() < self.block_factor: return None
        return x


class Batchify(Pipeline[I, Tp.Optional[Tp.List[I]]]):
    def __init__(self, input_type: Tp.Type, batch_size: int, omit_none=True):
        super().__init__()
        self.IN_TYPE = input_type
        self.OUT_TYPE = Tp.Optional[Tp.List[input_type]]
        self.batch_size = batch_size
        self.omit_none = omit_none
        self.buffer = []

    def __call__(self, entry: I) -> Tp.Optional[Tp.List[I]]:
        if self.omit_none and entry is None: return None

        self.buffer.append(entry)
        if len(self.buffer) == self.batch_size:
            batch = self.buffer[:]
            self.buffer = []
            return batch
        else:
            return None


class AutoBatchify(Pipeline[Tp.Optional[Tp.List[I]], Tp.Optional[Tp.List[O]]]):
    def __init__(self, pipe: Pipeline[I, O]):
        super().__init__()
        self.IN_TYPE = Tp.Optional[Tp.List[pipe.IN_TYPE]]
        self.OUT_TYPE = Tp.Optional[Tp.List[pipe.OUT_TYPE]]
        self.internal_pipe = pipe

    def __call__(self, batch):
        if batch is None: return None
        return [self.internal_pipe(entry) for entry in batch]

    def __repr__(self):
        return f"AutoBatchify({str(self.internal_pipe)})"


class ToSingletonSet(Pipeline[Tp.Any, Tp.Set[Tp.Any]]):
    def __init__(self, input_type):
        super().__init__()
        self.IN_TYPE = input_type
        self.OUT_TYPE = Tp.Set[input_type]

    def __call__(self, obj):
        if obj is None: return {}
        return {obj}


class ToSingletonList(Pipeline[Tp.Any, Tp.List[Tp.Any]]):
    def __init__(self, input_type):
        super().__init__()
        self.IN_TYPE = input_type
        self.OUT_TYPE = Tp.List[input_type]

    def __call__(self, obj):
        if obj is None: return []
        return [obj]
