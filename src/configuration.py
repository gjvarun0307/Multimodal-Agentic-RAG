"""
Centralized configuration management for the Multimodal Agentic RAG system.
Loads configuration from environment variables, JSON files, and provides sane defaults.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

API_KEYS_PATH = Path(__file__).resolve().parent.parent / "api_keys.json"

# Default model IDs are prefilled, editable placeholders for the Setup dashboard,
# not verified/hardcoded requirements -- check your provider's current model list.
LLM_PROVIDERS = {
    "anthropic": {"base_url": None, "default_model": "claude-haiku-4-5"},
    "openai": {"base_url": None, "default_model": "gpt-4o-mini"},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "default_model": "openai/gpt-oss-20b"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "default_model": "anthropic/claude-haiku-4.5"},
    "nvidia_nim": {"base_url": "https://integrate.api.nvidia.com/v1", "default_model": "meta/llama-3.1-8b-instruct"},
}


class Config:
    """Centralized configuration class."""

    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_configuration()
        return cls._instance

    def _load_configuration(self):
        """Load configuration from various sources."""
        # Start with defaults
        self._config = self._get_defaults()

        # Load from api_keys.json if it exists
        if API_KEYS_PATH.exists():
            try:
                with open(API_KEYS_PATH, 'r') as f:
                    api_keys = json.load(f)
                    self._config.update(api_keys)
            except Exception as e:
                print(f"Warning: Could not load api_keys.json: {e}")

        # Override with environment variables
        self._load_from_env()

    def reload(self):
        """Re-read api_keys.json and environment variables from scratch."""
        self._load_configuration()

    def _get_defaults(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            # API Keys (will be overridden by api_keys.json or env vars)
            "llama_parse": "",
            "tavilly_web": "",

            # Device configuration
            "device": "cuda" if torch.cuda.is_available() else "cpu",  # auto-detected; override via DEVICE env var

            # Database configuration
            "database_path": "./milvus.db",
            "input_folder_path": "data/raw_pdfs_parsed",

            # Text processing configuration
            "chunk_size": 1024,
            "overlap_size": 128,

            # Search configuration
            "sparse_weight": 0.7,
            "dense_weight": 1.0,
            "search_limit": 20,
            "reranker_score_threshold": 0.5,
            "reranker_top_k": 5,

            # Batch processing
            "insert_batch_size": 50,

            # Model configuration
            "embedding_model": "BAAI/bge-m3",
            "reranker_model": "BAAI/bge-reranker-v2-gemma",
            "use_fp16": False,
            "llm_provider": "",  # anthropic | openai | groq | openrouter | nvidia_nim -- set via Setup page
            "llm_model": "",
            "llm_api_key": "",
            "vision_model": "Qwen/Qwen2.5-VL-7B-Instruct",

            # Parsing configuration
            "parse_model_device": "cuda",
            "preload_parse_vlm": True,

            # Web search configuration
            "web_search_max_results": 5,
            "web_search_topic": "general",

            # Agent configuration
            "max_chat_turns": 10,
            "max_rewrites": 3,
            "max_gen_retries": 2,

            # Domains for routing
            "domain_topics": [
                "vLLM",
                "Transformers",
                "PagedAttention",
                "GraphRAG",
                "CRAG",
                "Self-RAG",
                "Adaptive-RAG",
                "Milvus",
                "HyDE",
                "LoRA",
                "LLaVA",
                "FlashAttention",
                "ColBERTv2",
            ]
        }

    def _load_from_env(self):
        """Load configuration from environment variables."""
        env_mapping = {
            # API Keys
            "LLAMA_PARSE_API_KEY": "llama_parse",
            "TAVILY_API_KEY": "tavilly_web",

            # Device
            "DEVICE": "device",

            # Database
            "DATABASE_PATH": "database_path",
            "INPUT_FOLDER_PATH": "input_folder_path",

            # Text processing
            "CHUNK_SIZE": ("chunk_size", int),
            "OVERLAP_SIZE": ("overlap_size", int),

            # Search weights
            "SPARSE_WEIGHT": ("sparse_weight", float),
            "DENSE_WEIGHT": ("dense_weight", float),
            "SEARCH_LIMIT": ("search_limit", int),
            "RERANKER_SCORE_THRESHOLD": ("reranker_score_threshold", float),
            "RERANKER_TOP_K": ("reranker_top_k", int),

            # Batch processing
            "INSERT_BATCH_SIZE": ("insert_batch_size", int),

            # Models
            "EMBEDDING_MODEL": "embedding_model",
            "RERANKER_MODEL": "reranker_model",
            "USE_FP16": ("use_fp16", lambda x: x.lower() == "true"),
            "LLM_PROVIDER": "llm_provider",
            "LLM_MODEL": "llm_model",
            "LLM_API_KEY": "llm_api_key",
            "VISION_MODEL": "vision_model",

            # Parsing
            "PARSE_MODEL_DEVICE": "parse_model_device",
            "PRELOAD_PARSE_VLM": ("preload_parse_vlm", lambda x: x.lower() == "true"),

            # Web search
            "WEB_SEARCH_MAX_RESULTS": ("web_search_max_results", int),
            "WEB_SEARCH_TOPIC": "web_search_topic",

            # Agent
            "MAX_CHAT_TURNS": ("max_chat_turns", int),
            "MAX_REWRITES": ("max_rewrites", int),
            "MAX_GEN_RETRIES": ("max_gen_retries", int),
        }

        for env_var, config_key in env_mapping.items():
            value = os.environ.get(env_var)
            if value is not None:
                # Handle type conversion if specified
                if isinstance(config_key, tuple):
                    key, converter = config_key
                    try:
                        self._config[key] = converter(value)
                    except (ValueError, TypeError) as e:
                        print(f"Warning: Could not convert {env_var}={value} to required type: {e}")
                else:
                    key = config_key
                    # Special handling for boolean-like strings
                    if key.endswith("_key") or "api" in key.lower():
                        self._config[key] = value
                    else:
                        self._config[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self._config.get(key, default)

    def set(self, key: str, value: Any):
        """Set a configuration value."""
        self._config[key] = value

    def as_dict(self) -> Dict[str, Any]:
        """Get all configuration as a dictionary."""
        return self._config.copy()

    def __getitem__(self, key: str) -> Any:
        return self._config[key]

    def __contains__(self, key: str) -> bool:
        return key in self._config


# Global config instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def reload_config() -> Config:
    """Re-read api_keys.json and environment variables into the existing singleton."""
    get_config().reload()
    return get_config()


def build_llm_client(config: Dict[str, Any]):
    """Construct a chat model client for the configured LLM provider.

    api_key is always passed explicitly (never left to provider SDKs' own
    env-var fallback, e.g. ANTHROPIC_API_KEY/OPENAI_API_KEY) so a stale
    shell-exported key can't silently override a dashboard-entered one.
    """
    provider = config.get("llm_provider", "")
    if provider not in LLM_PROVIDERS:
        raise ValueError(
            f"LLM provider not configured (got {provider!r}). "
            "Set it up on the Setup page, or via api_keys.json / the LLM_PROVIDER env var."
        )
    model = config.get("llm_model") or LLM_PROVIDERS[provider]["default_model"]
    api_key = config.get("llm_api_key", "")
    if provider == "anthropic":
        return ChatAnthropic(model=model, api_key=api_key)
    return ChatOpenAI(model=model, base_url=LLM_PROVIDERS[provider]["base_url"], api_key=api_key)


# Backward compatibility functions
def config_parse() -> Dict[str, Any]:
    """Backward compatibility function for existing code."""
    c = get_config()
    return {
        "api_key": c.get("llama_parse"),
        "input_folder": "data/raw_pdfs",
        "output_folder": "data/raw_pdfs_parsed",
        "device": c.get("device"),
        "prompt_imgcap": {
            "sys": "You are an expert Data Scientist and Technical Researcher. Your job is to analyze scientific figures, charts, and diagrams and extract structured data.",
            "user": """Analyze the provided image. Determine if it is a Chart (data visualization), a Diagram (architecture/process), or Irrelevant (logos, artifacts).

Return a valid JSON object with these fields:

title: The title of the figure (if visible) or a generated descriptive title.

type: 'chart', 'diagram', or 'irrelevant'.

features: A list of strings describing axis labels, legends, or key components (e.g., ['X-axis: Time', 'Y-axis: Latency']).

description: A detailed, technical paragraph explaining what the image shows. Focus on trends, relationships, and specific numbers if visible.

Output only the JSON."""
        },
        "prompt_metagen": {
            "sys": "You are a Research Librarian. Your task is to extract citation metadata from the first page of academic papers.",
            "user": """Look at this image of the first page of a research paper. Extract the following details into a strict JSON object:

title: The full title of the paper.

authors: A list of the author names.

year: The year of publication (search for dates near the top or bottom; if not found, estimate based on 'ArXiv' stamps or return 'Unknown').

topic: A one-phrase classification of the paper (e.g., 'Large Language Models', 'Vector Databases', 'Computer Vision').

Output only the JSON."""
        },
    }


def config_rag(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build the config dict consumed by the retrieval/generation pipeline.

    Precedence: Config defaults -> api_keys.json -> env vars (all resolved
    inside the Config singleton via _load_configuration()) -> `overrides`,
    applied last on top of the resolved dict. `overrides` lets a caller (the
    eval harness, an ablation runner) construct a fully config-driven run
    without touching env vars or api_keys.json.
    """
    c = get_config()
    resolved = {
        "device": c.get("device"),
        "tavilly_api_key": c.get("tavilly_web"),
        "database_path": c.get("database_path"),
        "input_folder_path": c.get("input_folder_path"),
        "chunk_size": c.get("chunk_size"),
        "overlap_size": c.get("overlap_size"),
        "domain_topics": c.get("domain_topics", []),
        "llm_provider": c.get("llm_provider"),
        "llm_model": c.get("llm_model"),
        "llm_api_key": c.get("llm_api_key"),
        "search_limit": c.get("search_limit"),
        "reranker_top_k": c.get("reranker_top_k"),
        "sparse_weight": c.get("sparse_weight"),
        "dense_weight": c.get("dense_weight"),
        "reranker_score_threshold": c.get("reranker_score_threshold"),
        "reranker_model": c.get("reranker_model"),
        "use_fp16": c.get("use_fp16"),
        "max_gen_retries": c.get("max_gen_retries"),
        "max_rewrites": c.get("max_rewrites"),
        "max_chat_turns": c.get("max_chat_turns"),
        "insert_batch_size": c.get("insert_batch_size"),
        "web_search_max_results": c.get("web_search_max_results"),
        "web_search_topic": c.get("web_search_topic"),
    }
    if overrides:
        resolved.update(overrides)
    return resolved
