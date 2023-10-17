"""
@brief: A parallel data processor that executes data pipeline
@author: Yutian Chen <yutianch@andrew.cmu.edu>
@date: May 16, 2023
"""
from typing import Callable, List, TypeVar
from pathlib import Path
from tqdm import tqdm

import copy
import multiprocessing as mp

T = TypeVar("T")


class PipelineExecutor:
    def __init__(self, worker_num):
        self.worker_num = mp.cpu_count() if (worker_num is None) else worker_num
        if self.worker_num > mp.cpu_count():
            print(f"You are using more multiprocess worker than cpu count ({mp.cpu_count()})!")

    def parallel_file_mapping(self, pipeline: Callable, from_files: List[Path], to_files: List[Path], write_mode="w", verbose=False, encoding=None):
        tqdm.set_lock(mp.RLock())
        assert len(from_files) == len(to_files)

        if verbose:
            print(f"PipelineExecutor: executing with {self.worker_num} workers")
            for from_file, to_file in zip(from_files, to_files):
                print(f"\t[{from_file}] => [{pipeline}]=> [{to_file}]")

        args = [(pipeline, from_file, to_file, write_mode, verbose, encoding) for from_file, to_file in zip(from_files, to_files)]

        with mp.Pool(self.worker_num, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as p:
            p.map(PipelineExecutor.file_mapping_worker, args)

        if verbose: print("Finish")

    @staticmethod
    def file_mapping_worker(args):
        pipeline, from_file, to_file, write_mode, verbose, encoding = args

        try: tqdm_position = mp.current_process()._identity[0] - 1
        except: tqdm_position = None

        with open(from_file, "r", encoding=encoding) as fin:
            with open(to_file, write_mode) as fout:
                try:
                    pb = tqdm(fin.readlines(), position=tqdm_position, leave=False) if verbose else fin.readlines()
                    for line in pb:
                        result = pipeline(line)
                        if result is not None: fout.write(result + "\n")
                finally:
                    fout.flush()

    def sequantial_file_mapping(self, pipeline: Callable, from_files: List[Path], to_files: List[Path], write_mode="w", verbose=False, encoding=None):
        assert len(from_files) == len(to_files)
        if verbose:
            print(f"PipelineExecutor: executing sequentially")
        for from_file, to_file in zip(from_files, to_files):
            if verbose: print(f"\t[{from_file}] --[{write_mode}]--> [{to_file}]")
            self.file_mapping_worker((pipeline, from_file, to_file, write_mode, verbose, encoding))

    def parallel_mapreduce(self, map_fn: Callable[[str], T], from_files: List[Path], identity: T, reduce_fn: Callable[[T, T], T], verbose=False, encoding=None) -> T:
        """
        :param map_fn: str (line in input file) -> 'a
        :param from_files: input files
        :param identity: 'a
        :param reduce_fn: ('a * 'a) -> 'a **Need to be associative**
        :return: The reduced result.
        """
        tqdm.set_lock(mp.RLock())
        if verbose:
            print(f"PipelineExecutor: mapreduce with {self.worker_num} workers")
            for from_file in from_files: print(f"\t[{from_file}] --[Map]--> [Reduce] -->")

        args = [(map_fn, from_file, identity, reduce_fn, verbose, encoding) for from_file in from_files]
        with mp.Pool(self.worker_num, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as p:
            reduce_results = p.map(PipelineExecutor.mapreduce_worker, args)

        final_reduction = identity
        for result in reduce_results:
            final_reduction = reduce_fn(final_reduction, result)

        return final_reduction

    @staticmethod
    def mapreduce_worker(args):
        map_fn, from_file, identity, reduce_fn, verbose = args

        try: tqdm_position = mp.current_process()._identity[0] - 1
        except: tqdm_position = None

        map_result = []
        with open(from_file, "r") as fin:
            pb = tqdm(fin.readlines(), position=tqdm_position, desc="Mapping") if verbose else fin.readlines()
            for line in pb: map_result.append(map_fn(line))

        reduce_result = identity
        pb = tqdm(map_result, position=tqdm_position, desc="Reducing") if verbose else map_result
        for item in pb:
            reduce_result = reduce_fn(reduce_result, item)

        return reduce_result

    def sequential_mapreduce(self, map_fn: Callable[[str], T], from_files: List[Path], identity: T, reduce_fn: Callable[[T, T], T], verbose=False) -> T:
        if verbose:
            print(f"PipelineExecutor: mapreduce sequentially")
            for from_file in from_files: print(f"\t[{from_file}] --[Map {map_fn}]--> [Reduce {reduce_fn}] -->")

        final_reduction = copy.deepcopy(identity)
        reduce_results = [self.mapreduce_worker((map_fn, from_file, copy.deepcopy(identity), reduce_fn, verbose))
                          for from_file in from_files]
        for result in reduce_results: final_reduction = reduce_fn(final_reduction, result)

        return final_reduction
