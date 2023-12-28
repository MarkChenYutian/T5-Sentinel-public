import yaml
from detector.t5_sentinel.types import Config


with open("detector/t5_sentinel/settings.yaml", "r") as f:
    config = Config(**yaml.safe_load(f))
