"""Serving Layer — dynamic micro-batching (RUNNABLE artifact).

Collects concurrent inference requests within a time window and processes them
in a single batch. This is the mechanism behind Triton's dynamic_batching{} /
vLLM continuous batching, implemented minimally so it is testable in-budget.
"""
from __future__ import annotations
import asyncio
from typing import Callable, Any


class DynamicBatcher:
    def __init__(self, batch_fn: Callable[[list[Any]], list[Any]],
                 max_batch: int = 8, max_delay_ms: int = 20):
        self.batch_fn = batch_fn
        self.max_batch = max_batch
        self.max_delay = max_delay_ms / 1000.0
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self.stats = {"batches": 0, "items": 0, "max_seen": 0}

    def start(self):
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def submit(self, item: Any) -> Any:
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        await self._queue.put((item, fut))
        return await fut

    async def _loop(self):
        while True:
            item, fut = await self._queue.get()
            batch = [(item, fut)]
            try:
                # wait up to max_delay collecting more, or until max_batch
                deadline = asyncio.get_event_loop().time() + self.max_delay
                while len(batch) < self.max_batch:
                    timeout = deadline - asyncio.get_event_loop().time()
                    if timeout <= 0:
                        break
                    try:
                        batch.append(await asyncio.wait_for(self._queue.get(), timeout))
                    except asyncio.TimeoutError:
                        break
                items = [b[0] for b in batch]
                self.stats["batches"] += 1
                self.stats["items"] += len(items)
                self.stats["max_seen"] = max(self.stats["max_seen"], len(items))
                results = await asyncio.to_thread(self.batch_fn, items)
                for (_, f), r in zip(batch, results):
                    if not f.done():
                        f.set_result(r)
            except Exception as e:  # noqa
                for _, f in batch:
                    if not f.done():
                        f.set_exception(e)
