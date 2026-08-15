"""vLLM provider entry (PROJECT_SPEC.md §4A) -- a measurement instrument,
not a deployed feature. Not always-on: launched on Colab/Kaggle behind a
tunnel for ablation runs, so its base_url must be a runtime override
rather than a fixed value in LLM_PROVIDERS.
"""

from langchain_openai import ChatOpenAI

from src.configuration import LLM_PROVIDERS, build_llm_client, config_rag


def test_vllm_provider_registered_with_overridable_base_url():
    assert "vllm" in LLM_PROVIDERS
    # None, not a fixed URL -- there's no stable endpoint to bake in here.
    assert LLM_PROVIDERS["vllm"]["base_url"] is None


def test_vllm_base_url_override_flows_into_client():
    resolved = config_rag(
        overrides={
            "llm_provider": "vllm",
            "llm_model": "meta-llama/Llama-3.1-8B-Instruct",
            "llm_api_key": "dummy",
            "vllm_base_url": "http://localhost:9999/v1",
        }
    )
    client = build_llm_client(resolved)

    assert isinstance(client, ChatOpenAI)
    assert client.openai_api_base == "http://localhost:9999/v1"
    assert client.model_name == "meta-llama/Llama-3.1-8B-Instruct"


def test_temperature_omitted_by_default_production_unaffected():
    # config_rag() with no overrides -- what app.py/src/api.py actually call
    # -- must never set a temperature the provider wouldn't have defaulted
    # to itself.
    resolved = config_rag(overrides={"llm_provider": "openai", "llm_model": "gpt-4o-mini", "llm_api_key": "dummy"})
    assert resolved["temperature"] is None
    client = build_llm_client(resolved)
    # langchain's default sentinel when temperature isn't passed at all
    assert client.temperature != 0


def test_temperature_override_flows_into_client():
    resolved = config_rag(
        overrides={"llm_provider": "openai", "llm_model": "gpt-4o-mini", "llm_api_key": "dummy", "temperature": 0}
    )
    client = build_llm_client(resolved)
    assert client.temperature == 0
