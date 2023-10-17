"""
@brief: An async generator used to collect OpenAI's classifier's response on test dataset
@author: Yutian Chen <yutianch@andrew.cmu.edu>
@date: May 16, 2023
"""
import asyncio
import aiohttp
import yaml
import json
import time

from typing import TypedDict, List, Tuple
from pathlib import Path
from generator.client_base import AsyncRequestClient, TaskResult
from pipeline.component.text_component import TextEntry
import pipeline.component.text_component as P

# Typing

class OpenAIState(TypedDict):
    processed: set


class OpenAIConfig(TypedDict):
    InputDirectory: List[str]
    OutputDirectory: List[str]
    WaitTime: float
    Header: dict
    URL: str

OpenAIArgs = Tuple[TextEntry, Path]
OpenAI_Type = AsyncRequestClient[OpenAIState, OpenAIArgs, OpenAIConfig]
###

load_data_fn = P.FromJsonStr() >> P.WriteExtra({"pred_by": "openai", "variant": "original"})

async def openai_request_fn(self: OpenAI_Type, state: OpenAIState, *args: OpenAIArgs) -> TaskResult:
    entry: TextEntry
    destination: Path
    entry, destination = args

    submission = {
        "model": "model-detect-v2",
        "max_tokens": 1, "temperature": 1, "top_p": 1, "n": 1, "logprobs": 5,
        "stop": "\n", "stream": False,
        "prompt": entry["text"] + "<|disc_score|>"
    }

    async with self.worker_lock:
        start_time = time.time()
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.post(self.config["URL"], headers=self.config["Header"], json=submission) as response:
                    status_code = response.status
                    result      = await response.json()

                    duration = time.time() - start_time
                    if status_code != 200:
                        await asyncio.sleep(self.config["WaitTime"] - duration)
                        return TaskResult.RETRY

                    async with self.writer_lock:
                        serializable = {
                            "uid": entry["uid"],
                            "extra": entry["extra"],
                            "res": result
                        }
                        with open(destination, "a", encoding="utf-8") as f: f.write(json.dumps(serializable) + "\n")

                    duration = time.time() - start_time
                    await asyncio.sleep(self.config["WaitTime"] - duration)

        except (aiohttp.ClientError, aiohttp.ServerTimeoutError, aiohttp.ServerDisconnectedError):
            await asyncio.sleep(self.config["WaitTime"])
            return TaskResult.RETRY

        except Exception as e:
            print("[x]\tUnexpected exception: ", e)
            return TaskResult.CANCEL

    return TaskResult.FINISH


def openai_pred_fn(client: OpenAI_Type, state: OpenAIState, *args: OpenAIArgs) -> bool:
    entry: TextEntry
    entry, dest = args
    return entry["uid"] not in state["processed"]


def openai_task_generator(client: OpenAI_Type, state: OpenAIState) -> List[OpenAIArgs]:
    Tasks = []
    for input_file, output_file in zip(client.config["InputDirectory"], client.config["OutputDirectory"]):
        counter = 0
        print(f"{input_file} --> {output_file}", end="\tCount:")
        assert Path(input_file).exists()
        with open(input_file, "r") as f:
            for line in f.read().strip().split("\n"):
                Tasks.append((load_data_fn(line), Path(output_file)))
                counter += 1
        print(counter)
    return Tasks


def openai_state_initializer(client: OpenAI_Type) -> OpenAIState:
    return {"processed": set()}


if __name__ == "__main__":
    with open("./detector/openai_classifier/openai_classifier_client.yaml", "r") as f:
        openai_config = yaml.safe_load(f)

    with open("./detector/openai_classifier/secret.json", "r") as f:
        openai_secret = json.load(f)
    openai_config["Config"]["Header"].update(openai_secret)

    OpenAIClient = OpenAI_Type(
        openai_config,
        openai_request_fn,
        openai_pred_fn,
        openai_task_generator,
        openai_state_initializer,
        display_args=lambda args: args[0]["uid"]
    )
    asyncio.run(OpenAIClient.execute())
