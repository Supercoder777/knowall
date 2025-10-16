import asyncio
import os
from importlib import reload
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def agent_module():
    os.environ["TEST_MODE"] = "true"
    import config

    reload(config)
    from modules import agent as agent_module

    reload(agent_module)
    return agent_module
def test_agent_test_mode(agent_module):
    trading_agent = agent_module.TradingAgent(Path("prompts"))
    result = asyncio.run(trading_agent.analyze_chart(b"binary", caption="EUR chart", metadata=None))
    assert result.pair == "EURUSD"
    asyncio.run(trading_agent.close())


def test_should_web_search_detects_keywords(agent_module):
    trading_agent = agent_module.TradingAgent(Path("prompts"))
    assert trading_agent._should_web_search("Need macro sentiment for GBPUSD") is True
    assert trading_agent._should_web_search("Pure technical view") is False
    asyncio.run(trading_agent.close())
