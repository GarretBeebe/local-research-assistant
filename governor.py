from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator

import psutil


class ResourceGovernor:
    """Limits concurrent researcher tasks and serializes under memory pressure."""

    def __init__(self, max_concurrent: int = 2, threshold_mb: int = 6144) -> None:
        self._max = max_concurrent
        self._threshold_mb = threshold_mb
        self._sem = asyncio.Semaphore(max_concurrent)
        self._pressure_lock = asyncio.Lock()

    def _under_pressure(self) -> bool:
        available_mb = psutil.virtual_memory().available >> 20
        return available_mb < self._threshold_mb or psutil.swap_memory().sin > 0

    @contextlib.asynccontextmanager
    async def slot(self) -> AsyncGenerator[None, None]:
        """Acquire a researcher slot, serializing if memory pressure is detected."""
        if self._under_pressure():
            async with self._pressure_lock:
                yield
        else:
            await self._sem.acquire()
            try:
                yield
            finally:
                self._sem.release()
