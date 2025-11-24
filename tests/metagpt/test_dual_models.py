import pytest
import asyncio
from metagpt.llm import LLM
from metagpt.config2 import Config

@pytest.mark.asyncio
async def test_dual_models():
    # Configuration for CoF-rl-model-7b (Port 8005)
    cof_config = Config.default()
    cof_config.llm.base_url = "http://localhost:8005/v1"
    cof_config.llm.model = "CoF-rl-model-7b"
    cof_config.llm.api_key = "EMPTY"
    
    # Configuration for Open-o3-Video (Port 8006)
    openo3_config = Config.default()
    openo3_config.llm.base_url = "http://localhost:8006/v1"
    openo3_config.llm.model = "Open-o3-Video"
    openo3_config.llm.api_key = "EMPTY"

    # Initialize LLMs
    print("\nInitializing CoF Model...")
    llm_cof = LLM(llm_config=cof_config.llm)
    
    print("Initializing Open-o3 Model...")
    llm_openo3 = LLM(llm_config=openo3_config.llm)

    # Test CoF Model
    print("\n--- Testing CoF-rl-model-7b ---")
    response_cof = await llm_cof.aask("Who are you?")
    print(f"CoF Response: {response_cof}")
    assert len(response_cof) > 0

    # Test Open-o3 Model
    print("\n--- Testing Open-o3-Video ---")
    response_openo3 = await llm_openo3.aask("Describe a video of a cat jumping.")
    print(f"Open-o3 Response: {response_openo3}")
    assert len(response_openo3) > 0

if __name__ == "__main__":
    asyncio.run(test_dual_models())
