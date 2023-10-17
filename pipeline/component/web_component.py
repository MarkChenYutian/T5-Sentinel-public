import time
import typing as Tp
from ..pipeline_base import Pipeline


I = Tp.TypeVar("I")


class RateControl(Pipeline[I, I]):
    def __init__(self, entry_per_min=60, omit_none=True, min_wait_time=0.0):
        super().__init__()
        self.entry_per_min = entry_per_min
        self.min_wait_time = min_wait_time
        self.omit_none = omit_none
        self.controller = {
            "start_t": 0.0,
            "count": 0
        }

    def __call__(self, entry):
        if entry is None and self.omit_none: return None

        now = time.time()
        if now - self.controller["start_t"] >= 60:
            self.controller["start_t"] = now
            self.controller["count"] = 0

        if self.controller["count"] >= self.entry_per_min:
            time.sleep(60.0 - (now - self.controller["start_t"]))
        else:
            time.sleep(self.min_wait_time)

        self.controller["count"] += 1
        return entry
