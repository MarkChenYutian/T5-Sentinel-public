"""
@brief: A Chat-GPT response generator using Async io
@author: Yutian Chen <yutianch@andrew.cmu.edu>
@date: March 19, 2023
"""

import asyncio
import random
import string
import openai
import yaml
import json
import time

import pipeline.component.text_component as P
from typing import TypedDict, List, Tuple
from pathlib import Path
from generator.client_base import AsyncRequestClient, TaskResult


# Typing

class ChatGPTState(TypedDict):
    processed: set
    token: int


class ChatGPTConfig(TypedDict):
    MaxTokenCount: int
    MaxLengthAllowed: int
    WaitTime: float
    InputDirectory: str
    OutputDirectory: str
    Sampling: float
    InputSubsets: List[str]


ChatGPTArgs = Tuple[str, str, str]

ChatGPTType = AsyncRequestClient[ChatGPTState, ChatGPTArgs, ChatGPTConfig]
Converter = P.WriteExtra({"source": "chatgpt", "variant": "original"}) >> P.ToJsonStr()
###


TOKEN_SPLITER = {char for char in string.punctuation + string.whitespace}
HANDLE_STRATEGY = {
    "stop": TaskResult.FINISH,
    "length": TaskResult.FINISH,
    "content_filter": TaskResult.CANCEL,
    "null": TaskResult.RETRY
}


def estimate_token_count(sample: str) -> int:
    est_num = 0
    for char in sample:
        est_num += 1 if char in TOKEN_SPLITER else 0
    return est_num


async def chatgpt_request_fn(self: ChatGPTType, state, subset, uid, text) -> TaskResult:
    if state["token"] > self.config["MaxTokenCount"]:
        print("Abort due to budget limit.")
        raise Exception("Exceed the MaxTokenCount setting")

    await self.worker_lock.acquire()
    start_time = time.time()

    # Ready ... now Work!

    estimatedNumTokens = estimate_token_count(text)
    if estimatedNumTokens > self.config["MaxLengthAllowed"]:
        print("[x]\t", uid,
              "failed since it exceeds the token limit (" + str(self.config["MaxLengthAllowed"]) + ")")
        self.worker_lock.release()
        return TaskResult.CANCEL

    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Rephrase the following paragraph by paragraph:\n\n" + text}
            ]
        )

    except openai.error.InvalidRequestError:
        # no need to wait, since the request is not sent for some reason
        await asyncio.sleep(1.0)  # Avoid flushing the API
        self.worker_lock.release()
        return TaskResult.RETRY

    except (openai.error.RateLimitError, openai.error.APIError, openai.error.TryAgain, openai.error.Timeout):
        await asyncio.sleep(self.config["WaitTime"])
        self.worker_lock.release()
        return TaskResult.RETRY

    finishReason = response["choices"][0]["finish_reason"]
    result = HANDLE_STRATEGY[finishReason]

    if result == TaskResult.FINISH:
        machineText = response["choices"][0]["message"]["content"].strip()

        await self.writer_lock.acquire()
        with open(Path(self.config["OutputDirectory"], subset + ".jsonl"), "a", encoding="utf-8") as f:
            f.write(Converter({"uid": uid, "text": machineText, "extra": dict()}))
            f.write("\n")
        self.writer_lock.release()
        self.state["processed"].add((subset, uid))

    self.state["token"] += response["usage"]["total_tokens"]

    # Wait for 60 secs, then release the lock to spawn a new worker coroutine
    # (We won't be blocked out)
    end_time = time.time()
    await asyncio.sleep(self.config["WaitTime"] - (end_time - start_time))
    self.worker_lock.release()

    return result


def chatgpt_pred_fn(client: ChatGPTType, state: ChatGPTState, subset, uid, text) -> bool:
    return (subset, uid) not in state["processed"]


def chatgpt_task_generator(client: ChatGPTType, state: ChatGPTState) -> List[ChatGPTArgs]:
    config = client.config
    task_args, subsets = [], config["InputSubsets"]

    for subset in subsets:
        print("Processing", subset)
        humanTextEntries = dict()

        with open(Path(config["InputDirectory"], subset + ".jsonl"), "r") as f:
            lines = f.read().strip().split("\n")
            for line in lines:
                entry = json.loads(line)
                humanTextEntries[entry["uid"]] = entry

        exist_count = len([uid for (_subset, uid) in state["processed"] if _subset == subset])
        target_count = int(len(humanTextEntries) * config["Sampling"])
        remain_cnt = max(target_count - exist_count, 0)
        remain_uids = random.choices(list(humanTextEntries.keys()), k=remain_cnt)

        for uid in remain_uids:
            task_args.append((subset, uid, humanTextEntries[uid]["text"]))

    return task_args


def chatgpt_state_initializer(client: ChatGPTType) -> ChatGPTState:
    processed = set()
    for subset in client.config["InputSubsets"]:
        if not Path(client.config["OutputDirectory"], subset + ".jsonl").exists(): continue
        with open(Path(client.config["OutputDirectory"], subset + ".jsonl"), "r") as f:
            lines = f.read().strip().split("\n")
            for line in lines:
                processed.add((subset, json.loads(line)["uid"]))

    return {"processed": processed, "token": 0}


def chatgpt_on_init_finish(client: ChatGPTType) -> None:
    ...


if __name__ == "__main__":
    with open("./generator/chatgpt/chatgpt_client.yaml", "r") as f:
        chatgpt_config = yaml.safe_load(f)

    with open(Path(chatgpt_config["ClientRoot"], "secret.json"), "r") as f:
        API_KEY = json.load(f)["OPENAI_API_KEY"]
        openai.api_key = API_KEY

    ChatGPTClient = AsyncRequestClient[ChatGPTState, ChatGPTArgs, ChatGPTConfig](
        chatgpt_config,
        chatgpt_request_fn,
        chatgpt_pred_fn,
        chatgpt_task_generator,
        chatgpt_state_initializer,
        on_init_finish=chatgpt_on_init_finish,
        display_args=lambda args: args[1]
    )
    asyncio.run(ChatGPTClient.execute())
