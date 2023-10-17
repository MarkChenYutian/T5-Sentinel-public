# Please set up the Google cloud credentials to use VertexAI service
# Link: https://cloud.google.com/vertex-ai/docs/tutorials/text-classification-automl

import os
from pathlib import Path
from typing import TypedDict

from google.cloud import aiplatform
import vertexai


class GCP_Config(TypedDict):
    project: str


def setup_credential(cred_path: Path, config: GCP_Config):
    os.putenv("GOOGLE_APPLICATION_CREDENTIALS", str(cred_path))
    vertexai.init(project=config["project"])
    if not Path("./generator/palm/gcp_init.lock").exists():
        aiplatform.init(project=config["project"])
        with open("./generator/palm/gcp_init.lock", "w") as f:
            f.write("Existence of this file shows that the google cloud is already initialized.")


if __name__ == "__main__":
    setup_credential(Path("./generator/palm/secret.json"), {"project": "llm-sentinel"})
