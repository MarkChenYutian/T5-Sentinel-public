import torch
import typing as Tp
from pathlib import Path

I = Tp.TypeVar("I")

def memoize(cache_path: Path, arg_eq):
    assert cache_path.parent.exists()

    def memoize_impl(func: Tp.Callable[[I], Tp.Any]):
        if not cache_path.exists():
            torch.save(dict(), cache_path)

        def wrapper(*args: I):
            result_dict = torch.load(cache_path)
            for prev_args in result_dict:
                if arg_eq(args, prev_args):
                    print(f"Reusing existing cache from {cache_path}")
                    return result_dict[prev_args]

            print("Cache Miss / Eviction since argument does not match")
            result = func(*args)
            result_dict[args] = result
            torch.save(result_dict, cache_path)
            return result
        return wrapper
    return memoize_impl

