import asyncio
import concurrent.futures
import functools
import inspect
import sys
import time
import typing
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Callable, Iterator, List, Optional, Set, TypeVar, cast

import synchronicity
from typing_extensions import ParamSpec
import atexit

from .logger import logger

synchronizer = synchronicity.Synchronizer()
atexit.register(synchronizer.close)


def synchronize_api(obj, target_module=None):
    if inspect.isclass(obj):
        blocking_name = obj.__name__.lstrip("_")
    elif inspect.isfunction(object):
        blocking_name = obj.__name__.lstrip("_")
    elif isinstance(obj, TypeVar):
        blocking_name = "_BLOCKING_" + obj.__name__
    else:
        blocking_name = None
    if target_module is None:
        target_module = obj.__module__
    return synchronizer.create_blocking(obj, blocking_name, target_module=target_module)


def retry(direct_fn=None, *, n_attempts=3, base_delay=0, delay_factor=2, timeout=90):
    def decorator(fn):
        @functools.wraps(fn)
        async def f_wrapped(*args, **kwargs):
            delay = base_delay
            for i in range(n_attempts):
                t0 = time.time()
                try:
                    return await asyncio.wait_for(fn(*args, **kwargs), timeout=timeout)
                except asyncio.CancelledError:
                    logger.debug(f"Function {fn} was cancelled")
                    raise
                except Exception as e:
                    if i >= n_attempts - 1:
                        raise
                    logger.debug(
                        f"Failed invoking function {fn}: {e}"
                        f" (took {time.time() - t0}s, sleeping {delay}s"
                        f" and trying {n_attempts - i - 1} more times)"
                    )
                await asyncio.sleep(delay)
                delay *= delay_factor

        return f_wrapped

    if direct_fn is not None:
        return decorator(direct_fn)
    else:
        return decorator


class TaskContext:
    _loops: Set[asyncio.Task]

    def __init__(self, grace: Optional[float] = None):
        self._grace = grace
        self._loops = set()

    async def start(self):
        self._tasks: set[asyncio.Task] = set()
        self._exited: asyncio.Event = asyncio.Event()

    @property
    def exited(self) -> bool:
        return self._exited.is_set()

    async def __aenter__(self):
        await self.start()
        return self
    async def stop(self):
        self._exited.set()
        await asyncio.sleep(0)
        unfinished_tasks = [t for t in self._tasks if not t.done()]
        gather_future = None
        try:
            if self._grace is not None and unfinished_tasks:
                gather_future = asyncio.gather(*unfinished_tasks, return_exceptions=True)
                await asyncio.wait_for(gather_future, timeout=self._grace)
        except asyncio.TimeoutError:
            pass
        finally:
            if gather_future:
                try:
                    await gather_future
                except asyncio.CancelledError:
                    pass

            for task in self._tasks:
                if task.done() and not task.cancelled():
                    task.result()

                if task.done() or task in self._loops:
                    continue

                if sys.version_info >= (3, 11):
                    already_cancelling = task.cancelling() > 0
                    if not already_cancelling:
                        logger.warning(f"Canceling remaining unfinished task: {task}")

                task.cancel()

    async def __aexit__(self, exc_type, value, tb):
        await self.stop()

    def create_task(self, coro_or_task) -> asyncio.Task:
        if isinstance(coro_or_task, asyncio.Task):
            task = coro_or_task
        elif asyncio.iscoroutine(coro_or_task):
            loop = asyncio.get_event_loop()
            task = loop.create_task(coro_or_task)
        else:
            raise Exception(f"Object of type {type(coro_or_task)} is not a coroutine or Task")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def infinite_loop(self, async_f, timeout: Optional[float] = 90, sleep: float = 10) -> asyncio.Task:
        function_name = async_f.__qualname__

        async def loop_coro() -> None:
            logger.debug(f"Starting infinite loop {function_name}")
            while True:
                t0 = time.time()
                try:
                    await asyncio.wait_for(async_f(), timeout=timeout)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    time_elapsed = time.time() - t0
                    logger.exception(f"Loop attempt failed for {function_name} (time_elapsed={time_elapsed})")
                try:
                    await asyncio.wait_for(self._exited.wait(), timeout=sleep)
                except asyncio.TimeoutError:
                    continue
                logger.debug(f"Exiting infinite loop for {function_name}")
                break

        t = self.create_task(loop_coro())
        t.set_name(f"{function_name} loop")
        self._loops.add(t)
        t.add_done_callback(self._loops.discard)
        return t

    async def wait(self, *tasks):
        unfinished_tasks = set(tasks)
        while True:
            unfinished_tasks &= self._tasks
            if not unfinished_tasks:
                break
            try:
                done, pending = await asyncio.wait_for(
                    asyncio.wait(self._tasks, return_when=asyncio.FIRST_COMPLETED), timeout=30.0
                )
            except asyncio.TimeoutError:
                continue
            for task in done:
                task.result()
                if task in unfinished_tasks:
                    unfinished_tasks.remove(task)
                if task in self._tasks:
                    self._tasks.remove(task)


def run_coro_blocking(coro):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        fut = executor.submit(asyncio.run, coro)
        return fut.result()


async def queue_batch_iterator(q: asyncio.Queue, max_batch_size=100, debounce_time=0.015):
    item_list: List[Any] = []

    while True:
        if q.empty() and len(item_list) > 0:
            yield item_list
            item_list = []
            await asyncio.sleep(debounce_time)

        res = await q.get()

        if len(item_list) >= max_batch_size:
            yield item_list
            item_list = []

        if res is None:
            if len(item_list) > 0:
                yield item_list
            break
        item_list.append(res)


class _WarnIfGeneratorIsNotConsumed:
    def __init__(self, gen, gen_f):
        self.gen = gen
        self.gen_f = gen_f
        self.iterated = False
        self.warned = False

    def __aiter__(self):
        self.iterated = True
        return self.gen

    async def __anext__(self):
        self.iterated = True
        return await self.gen.__anext__()

    def __repr__(self):
        return repr(self.gen)

    def __del__(self):
        if not self.iterated and not self.warned:
            self.warned = True
            name = self.gen_f.__name__
            logger.warning(
                f"Warning: the results of a call to {name} was not consumed, so the call will never be executed."
                f" Consider a for-loop like `for x in {name}(...)` or unpacking the generator using `list(...)`"
            )


synchronize_api(_WarnIfGeneratorIsNotConsumed)


def warn_if_generator_is_not_consumed(gen_f):
    @functools.wraps(gen_f)
    def f_wrapped(*args, **kwargs):
        gen = gen_f(*args, **kwargs)
        return _WarnIfGeneratorIsNotConsumed(gen, gen_f)

    return f_wrapped


_shutdown_tasks = []


def on_shutdown(coro):
    async def wrapper():
        try:
            await asyncio.sleep(1e10)
        finally:
            await coro
            raise

    _shutdown_tasks.append(asyncio.create_task(wrapper()))


T = TypeVar("T")
P = ParamSpec("P")


def asyncify(f: Callable[P, T]) -> Callable[P, typing.Coroutine[None, None, T]]:
    @functools.wraps(f)
    async def wrapper(*args: P.args, **kwargs: P.kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, functools.partial(f, *args, **kwargs))

    return wrapper


async def iterate_blocking(iterator: Iterator[T]) -> AsyncGenerator[T, None]:
    loop = asyncio.get_running_loop()
    DONE = object()
    while True:
        obj = await loop.run_in_executor(None, next, iterator, DONE)
        if obj is DONE:
            break
        yield cast(T, obj)


class ConcurrencyPool:
    def __init__(self, concurrency_limit: int):
        self.semaphore = asyncio.Semaphore(concurrency_limit)

    async def run_coros(self, coros: typing.Iterable[typing.Coroutine], return_exceptions=False):
        async def blocking_wrapper(coro):
            try:
                await self.semaphore.acquire()
            except asyncio.CancelledError:
                coro.close()

            try:
                res = await coro
                self.semaphore.release()
                return res
            except BaseException as e:
                if return_exceptions:
                    self.semaphore.release()
                raise e

        tasks = [asyncio.create_task(blocking_wrapper(coro)) for coro in coros]
        g = asyncio.gather(*tasks, return_exceptions=return_exceptions)
        try:
            return await g
        except BaseException as e:
            for t in tasks:
                t.cancel()
            raise e

@asynccontextmanager
async def asyncnullcontext(*args, **kwargs):
    yield
