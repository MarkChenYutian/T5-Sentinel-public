"""
@brief: An async generator used to collect ZeroGPT's classifier's response on test dataset
@author: Yutian Chen <yutianch@andrew.cmu.edu>
@date: June 10, 2023
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

class ZeroGPTState(TypedDict):
    processed: set


class ZeroGPTConfig(TypedDict):
    InputDirectory: List[str]
    OutputDirectory: List[str]
    WaitTime: float
    Header: dict
    URL: str

ZeroGPTArgs = Tuple[TextEntry, Path]
ZeroGPT_Client = AsyncRequestClient[ZeroGPTState, ZeroGPTArgs, ZeroGPTConfig]
###

load_data_fn = P.FromJsonStr() >> P.WriteExtra({"pred_by": "zerogpt", "variant": "original"})

async def zerogpt_request_fn(self: ZeroGPT_Client, state: ZeroGPTState, *args: ZeroGPTArgs) -> TaskResult:
    entry: TextEntry
    destination: Path
    entry, destination = args

    submission = {"input_text": entry["text"]}

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
            print("[x]\tClientError | ServerTimeoutError | ServerDisconnectedError: ")
            await asyncio.sleep(self.config["WaitTime"])
            return TaskResult.RETRY

        except Exception as e:
            print("[x]\tUnexpected exception: ", e)
            await asyncio.sleep(self.config["WaitTime"])
            return TaskResult.RETRY

    return TaskResult.FINISH


def zerogpt_pred_fn(client: ZeroGPT_Client, state: ZeroGPTState, *args: ZeroGPTArgs) -> bool:
    entry: TextEntry
    entry, dest = args
    return entry["uid"] not in state["processed"]


def zerogpt_task_generator(client: ZeroGPT_Client, state: ZeroGPTState) -> List[ZeroGPTArgs]:
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


def zerogpt_state_initializer(client: ZeroGPT_Client) -> ZeroGPTState:
    return {"processed": set()}


if __name__ == "__main__":
    with open("./detector/zerogpt_classifier/zerogpt_classifier_client.yaml", "r") as f:
        openai_config = yaml.safe_load(f)

    ZeroGPTClient = ZeroGPT_Client(
        openai_config,
        zerogpt_request_fn,
        zerogpt_pred_fn,
        zerogpt_task_generator,
        zerogpt_state_initializer,
        display_args=lambda args: args[0]["uid"]
    )
    asyncio.run(ZeroGPTClient.execute())
