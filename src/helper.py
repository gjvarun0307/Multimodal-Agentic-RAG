import json
import re
from typing import Optional

import openai


def is_rate_limit_error(exc: BaseException) -> bool:
    """True if exc (or anything in its __cause__/__context__ chain, e.g. a
    JudgeGradingError wrapping the real provider exception) is an
    openai.RateLimitError -- every LLM provider in this project, including
    Groq, is called through langchain_openai.ChatOpenAI (see
    src/configuration.py's build_llm), so a 429 always surfaces as this
    exact exception type regardless of which provider is configured.
    Distinguishes a quota blip from a real regression (CLAUDE.md Phase 4
    checklist) rather than lumping both into one generic fallback tag."""
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, openai.RateLimitError):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


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
