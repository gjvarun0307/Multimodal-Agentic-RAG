import re
import json
from pathlib import Path
import sys

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
