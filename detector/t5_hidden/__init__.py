import yaml
from detector.t5_hidden.types import Config


with open("detector/t5_hidden/settings.yaml", "r") as f:
    config = Config(**yaml.safe_load(f))
