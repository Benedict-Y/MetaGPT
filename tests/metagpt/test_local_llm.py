import pytest
from metagpt.llm import LLM

@pytest.mark.asyncio
async def test_local_llm():
    llm = LLM()
    response = await llm.aask("Hello, are you working?")
    print(f"\nResponse from LLM: {response}")
    assert len(response) > 0

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_local_llm())
