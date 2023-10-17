from pipeline import P, PipelineExecutor
from pathlib import Path


def sanitize(from_subset, to_subset):
    clean_pipeline = P.FromJsonStr() \
                     >> P.WriteExtra({"variant": "sanitized"}) \
                     >> P.StripNewline() \
                     >> P.CastUnicode() \
                     >> P.RemovePunc() \
                     >> P.ToLower() \
                     >> P.RemoveContSpace() \
                     >> P.ToJsonStr()

    executor = PipelineExecutor(worker_num=None)
    executor.parallel_file_mapping(
        clean_pipeline,
        from_files=[Path("./data/split/", subset) for subset in from_subset],
        to_files=[Path("./data/split/", subset) for subset in to_subset],
        verbose=True
    )

