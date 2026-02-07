import re

# json cleaning for parse json from model
def clean_json_text(text):
    text = text.strip()
    # Remove markdown code blocks if present
    if text.startswith("```"):
        text = re.sub(r"^```(json)?\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return text