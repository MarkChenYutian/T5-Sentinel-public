import typing as Tp

from pathlib import Path
from ..component import Pipeline

I = Tp.TypeVar("I")


class Print(Pipeline[I, I]):
    def __init__(self, prefix: Tp.Optional[str]=None, omit_none=False):
        super().__init__()
        self.prefix = "" if prefix is None else prefix
        self.omit_none = omit_none

    def __call__(self, obj):
        if self.omit_none:
            if obj is not None: print(self.prefix, obj)
        else: print(self.prefix, obj)
        return obj


class WriteTo(Pipeline[Tp.Optional[str], None]):
    def __init__(self, destination: Path, write_mode="a"):
        super().__init__()
        assert destination.parent.exists(), f"Writing to {destination} but parent does not exist"
        self.destination = destination
        self.write_mode = "a"

    def __call__(self, string):
        if string is None: return
        with open(self.destination, self.write_mode) as f:
            f.write(string + "\n")

