"""
@brief: A generic async generator
@author: Yutian Chen <yutianch@andrew.cmu.edu>
@date: March 19, 2023
"""
import time
import pickle
import asyncio

from typing import Generic, TypeVar, Callable, List, Awaitable, Any, Optional
from enum import Enum
from pathlib import Path


class TaskResult(Enum):
    CANCEL = 2
    RETRY = 1
    FINISH = 0


S = TypeVar("S")    # State type
A = TypeVar("A")    # Task_arg type
C = TypeVar("C")    # Config type


class AsyncRequestClient(Generic[S, A, C]):
    def __init__(self,
                 config: C,
                 request_fn: Callable[["AsyncRequestClient", S, A], Awaitable[TaskResult]],
                 pred_fn: Callable[["AsyncRequestClient", S, A], bool],
                 task_generator: Callable[["AsyncRequestClient", S], List[A]],
                 state_initializer: Callable[["AsyncRequestClient"], S],
                 on_init_finish: Optional[Callable[["AsyncRequestClient"], Any]]=None,
                 on_dump_state: Optional[Callable[["AsyncRequestClient"], Any]]=None,
                 on_task_fail: Optional[Callable[["AsyncRequestClient", S, A], Any]]=None,
                 display_args: Optional[Callable[[A], Any]]=None
                 ):
        """
        Initialize an async generator

        :param request_fn: Async function, Send request, parse result and report the RequestResult
            (generator, state, *task_args) -> AsyncRequestResult

        :param pred_fn: function, given task arguments, return if this task is valid
            - return true => the task will be executed
            - return false => the task will be skipped
            (generator, state, *task_args) -> bool

        :param task_generator: function, given state, generate a list of task
            (generator, state) -> List[task_args]

        :param state_initializer: create an empty state
            (generator,) -> state

        :param on_init_finish: execute the side effects when generator's __init__ function finishes
            (generator,) -> Any

        :param display_args: visualize the argument of task
            (task_args) -> Any

        :param config:
            some dictionary
        """
        self.client_name = config["ClientName"]
        self.client_root = Path(config["ClientRoot"])
        self.MAX_ASYNC_WORKER_CNT = config["MaxAsyncWorkerCnt"]
        self.MAX_RETRY_CNT = config["MaxRetryCnt"]
        self.config: C = config["Config"]

        # Setup callback functions
        self.request_fn = request_fn
        self.pred_fn = pred_fn
        self.task_generator = task_generator
        self.state_initializer = state_initializer
        self.on_dump_state = on_dump_state
        self.on_task_fail = on_task_fail
        self.display_args: Callable[[A], Any] = display_args if display_args is not None else (lambda x : x)

        # Initialize the generator
        self.state: S = self._load_client_state()

        # async_status
        self.worker_lock: asyncio.Semaphore  # Semaphore to control request rate
        self.writer_lock: asyncio.Semaphore  # Mutex to avoid data racing
        self.finish_task = 0
        self.total_task = 0
        self.canceled = False

        if on_init_finish is not None: on_init_finish(self)

    async def task_worker(self, *args: A) -> None:
        try:
            finish_flag = False
            for try_cnt in range(self.MAX_RETRY_CNT):
                status: TaskResult = await self.request_fn(self, self.state, *args)
                if status == TaskResult.FINISH:
                    self.finish_task += 1
                    print(f"\t[{self.finish_task} / {self.total_task}] \t- FINISH {self.display_args(args)}")
                    finish_flag = True
                    break
                elif status == TaskResult.RETRY:
                    print(f"[-]\t[{self.finish_task} / {self.total_task}] \t - RETRY ({try_cnt + 1} / {self.MAX_RETRY_CNT})")
                    print(f"\t\tRetry - {self.display_args(args)}")
                elif status == TaskResult.CANCEL:
                    self.finish_task += 1
                    if self.on_task_fail is not None: self.on_task_fail(self, self.state, args)
                    print(f"[x]\t[{self.finish_task} / {self.total_task}] \t - CANCEL {self.display_args(args)}")
                    finish_flag = True
                    break

            if not finish_flag:
                self.finish_task += 1
                if self.on_task_fail is not None: self.on_task_fail(self, self.state, args)
                print(f"[x]\t[{self.finish_task} / {self.total_task}] \t - CANCEL after trying {self.MAX_RETRY_CNT} times. {self.display_args(args)}")
        except (asyncio.CancelledError, KeyboardInterrupt) as e:
            if not self.canceled:
                print("All tasks cancelled by asyncio")
                self.canceled = True
            raise e

    async def execute(self) -> None:
        print("Creating tasks ... ")
        self.worker_lock = asyncio.Semaphore(self.MAX_ASYNC_WORKER_CNT)
        self.writer_lock = asyncio.Semaphore(1)

        tasks_args = [
            task for task in self.task_generator(self, self.state)
            if self.pred_fn(self, self.state, *task)
        ]
        self.total_task = len(tasks_args)
        print(f"{self.total_task} tasks created.")

        try:
            tasks = [
                asyncio.create_task(self.task_worker(*task_arg))
                for task_arg in tasks_args
            ]
            await asyncio.gather(*tasks)
            print("All tasks complete. Exit.")
        except Exception as e:
            print("Client interrupted by an exception, saving internal state ...")
            print("Exit.")
            raise e
        finally:
            self._dump_client_state()

    def _load_client_state(self) -> S:
        if not Path(self.client_root, f"{self.client_name}_state.pkl").exists():
            return self.state_initializer(self)
        with open(Path(self.client_root, f"{self.client_name}_state.pkl"), "rb") as f:
            return pickle.load(f)

    def _dump_client_state(self) -> None:
        if self.on_dump_state is not None: self.on_dump_state(self)

        with open(Path(self.client_root, f"{self.client_name}_state.pkl"), "wb") as f:
            pickle.dump(self.state, f)


class SyncRequestClient(Generic[S, A, C]):
    def __init__(self,
                 config,
                 request_fn: Callable[["SyncRequestClient", S, A], TaskResult],
                 pred_fn: Callable[["SyncRequestClient", S, A], bool],
                 task_generator: Callable[["SyncRequestClient", S], List[A]],
                 state_initializer: Callable[["SyncRequestClient"], S],
                 on_init_finish: Optional[Callable[["SyncRequestClient"], Any]] = None,
                 on_dump_state : Optional[Callable[["SyncRequestClient"], Any]] = None,
                 on_task_fail  : Optional[Callable[["SyncRequestClient", S, A], Any]] = None,
                 display_args: Optional[Callable[[A], Any]] = None
                 ):
        self.RATE_LIMIT = config["RateLimit"]
        self.MAX_RETRY_CNT = config["MaxRetryCnt"]
        self.client_name = config["ClientName"]
        self.client_root = Path(config["ClientRoot"])
        self.config: C = config["Config"]

        # Setup callback functions
        self.request_fn = request_fn
        self.pred_fn = pred_fn
        self.task_generator = task_generator
        self.state_initializer = state_initializer
        self.on_dump_state = on_dump_state
        self.on_task_fail = on_task_fail
        self.display_args: Callable[[A], Any] = display_args if display_args is not None else (lambda x: x)

        # Initialize the generator
        self.state: S = self._load_client_state()

        self.finish_task = 0
        self.total_task = 0
        self.rate_controller = {"time": time.time(), "count": 0}

        on_init_finish(self)

    def task_worker(self, *args: A):
        try:
            current_time = time.time()
            if current_time - self.rate_controller["time"] > 60:
                self.rate_controller = {"time": current_time, "count": 0}
            if self.rate_controller["count"] == self.RATE_LIMIT:
                duration = (self.rate_controller["time"] + 60 - current_time)
                print(f"[-]\tRate Controller: Sleep for {duration} to control task rate")
                time.sleep(duration)

            finish_flag = False
            for try_cnt in range(self.MAX_RETRY_CNT):
                status: TaskResult = self.request_fn(self, self.state, *args)
                if status == TaskResult.FINISH:
                    self.finish_task += 1
                    print(f"\t[{self.finish_task} / {self.total_task}] \t - FINISH {self.display_args(args)}")
                    finish_flag = True
                    break
                elif status == TaskResult.RETRY:
                    print(f"[-]\t[{self.finish_task} / {self.total_task}] \t - RETRY ({try_cnt + 1} / {self.MAX_RETRY_CNT})"
                          f"{self.display_args(args)}")
                elif status == TaskResult.CANCEL:
                    self.finish_task += 1
                    if self.on_task_fail is not None: self.on_task_fail(self, self.state, args)
                    print(f"[x]\t[{self.finish_task} / {self.total_task}] \t - CANCEL {self.display_args(args)}")
                    finish_flag = True
                    break

            if not finish_flag:
                self.finish_task += 1
                if self.on_task_fail is not None: self.on_task_fail(self, self.state, args)
                print(f"[x]\t[{self.finish_task} / {self.total_task}] \t - CANCEL after trying {self.MAX_RETRY_CNT} times. {self.display_args(args)}")

        except Exception as e:
            print(f"Unexpected exception raised: {e}. Abort.")
            self._dump_client_state()
            raise e


    def execute(self):
        print("Creating tasks ... ")

        tasks_args = [
            task for task in self.task_generator(self, self.state)
            if self.pred_fn(self, self.state, *task)
        ]
        self.total_task = len(tasks_args)
        print(f"{self.total_task} tasks created.")

        try:
            tasks = [self.task_worker(*task_arg) for task_arg in tasks_args]
            print("All tasks complete. Exit.")
        except Exception as e:
            print("Client interrupted by an exception, saving internal state ...")
            print("Exit.")
            raise e
        finally:
            self._dump_client_state()

    def _load_client_state(self) -> S:
        if not Path(self.client_root, f"{self.client_name}_state.pkl").exists():
            return self.state_initializer(self)
        with open(Path(self.client_root, f"{self.client_name}_state.pkl"), "rb") as f:
            return pickle.load(f)

    def _dump_client_state(self) -> None:
        if self.on_dump_state is not None: self.on_dump_state(self)

        with open(Path(self.client_root, f"{self.client_name}_state.pkl"), "wb") as f:
            pickle.dump(self.state, f)
