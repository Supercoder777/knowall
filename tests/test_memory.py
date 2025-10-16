import asyncio
from pathlib import Path

import numpy as np

from tools.memory import EmbeddingMemory


async def _exercise(memory: EmbeddingMemory) -> None:
    vector = np.ones(8, dtype="float32")
    await memory.add_entry("test summary", vector, {"pair": "GBPUSD"})
    results = await memory.search(vector, top_k=1)
    assert results
    assert results[0].metadata["pair"] == "GBPUSD"


def test_memory_add_and_search(tmp_path: Path) -> None:
    storage = tmp_path / "mem"
    memory = EmbeddingMemory(storage, dimension=8)
    asyncio.run(_exercise(memory))
