from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator

import psutil


class ResourceGovernor:
    """Limits concurrent researcher tasks and serializes under memory pressure."""

    def __init__(self, max_concurrent: int = 2, threshold_mb: int = 6144) -> None:
        if max_concurrent < 1:
            raise ValueError(f"max_concurrent must be >= 1, got {max_concurrent}")
        self._max = max_concurrent
        self._threshold_mb = threshold_mb
        self._sem = asyncio.Semaphore(max_concurrent)
        # Serializes multi-slot drains under pressure to prevent deadlock between
        # two pressure-path tasks each waiting for the other's semaphore slot.
        self._drain_lock = asyncio.Lock()

    def _under_pressure(self) -> bool:
        return psutil.virtual_memory().available >> 20 < self._threshold_mb

    @contextlib.asynccontextmanager
    async def slot(self) -> AsyncGenerator[None, None]:
        """Acquire a researcher slot, serializing under memory pressure.

        Under pressure: drains all semaphore slots before yielding so only one task
        runs at a time. Normal: acquires one slot from the shared semaphore pool.
        Both paths use _sem as the single concurrency gate.
        """
        if self._under_pressure():
            async with self._drain_lock:
                for _ in range(self._max):
                    await self._sem.acquire()
            try:
                yield
            finally:
                for _ in range(self._max):
                    self._sem.release()
        else:
            await self._sem.acquire()
            try:
                yield
            finally:
                self._sem.release()
