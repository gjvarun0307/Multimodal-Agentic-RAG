import json

import streamlit as st

from src.configuration import (
    API_KEYS_PATH,
    LLM_PROVIDERS,
    build_llm_client,
    get_config,
    reload_config,
)

st.set_page_config(page_title="Setup — Multimodal Agentic RAG", page_icon="⚙️")
st.title("Setup")
st.caption(
    "Configure your own API keys. They're written to a gitignored api_keys.json "
    "in the project root and never leave this machine."
)

config = get_config()


def _status(value, label):
    if value:
        st.success(f"{label}: configured")
    else:
        st.warning(f"{label}: missing")


st.subheader("LLM Provider")

provider_names = list(LLM_PROVIDERS.keys())
current_provider = config.get("llm_provider", "")
default_index = provider_names.index(current_provider) if current_provider in provider_names else 0
provider = st.selectbox("Provider", provider_names, index=default_index)

# Keying on the provider means switching providers resets this field to that
# provider's default instead of carrying over a mismatched model name.
model = st.text_input(
    "Model",
    value=config.get("llm_model") or LLM_PROVIDERS[provider]["default_model"],
    key=f"model_{provider}",
)

llm_key = st.text_input("API key", type="password", value="", key=f"llm_key_{provider}")
_status(config.get("llm_api_key"), "Current LLM key")

if provider == "openrouter":
    st.caption(
        "OpenRouter model choice matters: tool-calling support varies by the "
        "underlying model, and this app's routing/rewriting/grading nodes depend "
        "on it working. Prefer well-known tool-calling-capable models."
    )

if st.button("Test LLM connection"):
    test_config = {
        **config.as_dict(),
        "llm_provider": provider,
        "llm_model": model,
        "llm_api_key": llm_key or config.get("llm_api_key"),
    }
    try:
        client = build_llm_client(test_config)
        response = client.invoke("Reply with the single word: ok")
        st.success(f"Connected. Response: {str(response.content)[:200]}")
    except Exception as e:
        st.error(f"Connection failed: {e}")

st.divider()

st.subheader("LlamaParse")
llama_key = st.text_input("LlamaParse API key", type="password", value="")
_status(config.get("llama_parse"), "Current LlamaParse key")

st.divider()

st.subheader("Tavily (web search)")
tavily_key = st.text_input("Tavily API key", type="password", value="")
_status(config.get("tavilly_web"), "Current Tavily key")

st.divider()

if st.button("Save", type="primary"):
    existing = {}
    if API_KEYS_PATH.exists():
        try:
            existing = json.loads(API_KEYS_PATH.read_text())
        except Exception as e:
            st.error(f"Could not read existing api_keys.json: {e}")
            st.stop()

    existing["llm_provider"] = provider
    existing["llm_model"] = model
    if llm_key:
        existing["llm_api_key"] = llm_key
    if llama_key:
        existing["llama_parse"] = llama_key
    if tavily_key:
        existing["tavilly_web"] = tavily_key

    API_KEYS_PATH.write_text(json.dumps(existing, indent=2))
    reload_config()
    st.cache_resource.clear()
    st.success("Saved. Reloading with the new configuration.")
    st.rerun()
