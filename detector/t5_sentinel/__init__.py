import yaml
from detector.t5_sentinel.types import Config


with open('detector/models/arbitrary/settings.yaml', 'r') as f:
    config = Config(**yaml.safe_load(f))
