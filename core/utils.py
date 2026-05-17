import time


async def timed(coro) -> tuple[list, float]:
    """Run a coroutine and return (result, elapsed_seconds)."""
    t0 = time.monotonic()
    result = await coro
    return result, time.monotonic() - t0
