import re
import json
from pathlib import Path
import sys

from hybrid_database import load_database_and_embedding
from FlagEmbedding import FlagLLMReranker
from langchain_openai import ChatOpenAI

# json cleaning for parse json from model
def clean_json_text(text):
    text = text.strip()
    # Remove markdown code blocks if present
    if text.startswith("```"):
        text = re.sub(r"^```(json)?\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return text

def open_jsonl(path_to_jsonl):
    """
    opens jsonl files and return list of json
    
    :param path_to_jsonl: path to jsonl file
    """
    with open(path_to_jsonl, 'r') as json_file:
        json_list = []
        lines = json_file.readlines()
    for line in lines:
        line = json.loads(line)
        json_list.append(line)

    return json_list

def get_models(config):
    
    database, embedding_model = load_database_and_embedding(database_path=config["database_path"], device=config["device"])
    print("loaded datbase and embedding_model")

    rerank_model = FlagLLMReranker('BAAI/bge-reranker-v2-gemma', use_fp16=True, devices=config["device"])
    print("loaded rerank model")

    llm_model = ChatOpenAI(
        model_name="Qwen/Qwen3-8B-AWQ",
        base_url="http://localhost:8000/v1",
        api_key="token-abc123",
        model_kwargs={
            "extra_body": {
                "guided_decoding_backend": "xgrammar" 
            }
        }
    )
    print("loaded llm model")
    print("All models loaded!!")
    return database, embedding_model, rerank_model, llm_model