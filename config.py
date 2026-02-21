import os
import json


def config_parse():
    with open("api_keys.json") as f:
        api_keys = json.load(f)
    return {
        "api_key": api_keys['llama_parse'],
        "input_folder": "data/test_pdfs",
        "output_folder": "data/test_pdf_parsed",
        # model device for transformers
        "device": 'cuda',
        # prompt for vlm
        "prompt_imgcap": {
            "sys": "You are an expert Data Scientist and Technical Researcher. Your job is to analyze scientific figures, charts, and diagrams and extract structured data.",
            "user": """Analyze the provided image. Determine if it is a Chart (data visualization), a Diagram (architecture/process), or Irrelevant (logos, artifacts).\n\nReturn a valid JSON object with these fields:\n\ntitle: The title of the figure (if visible) or a generated descriptive title.\n\ntype: 'chart', 'diagram', or 'irrelevant'.\n\nfeatures: A list of strings describing axis labels, legends, or key components (e.g., ['X-axis: Time', 'Y-axis: Latency']).\n\ndescription: A detailed, technical paragraph explaining what the image shows. Focus on trends, relationships, and specific numbers if visible.\n\nOutput only the JSON."""
        },
        "prompt_metagen": {
            "sys": "You are a Research Librarian. Your task is to extract citation metadata from the first page of academic papers.",
            "user": """Look at this image of the first page of a research paper. Extract the following details into a strict JSON object:\n\ntitle: The full title of the paper.\n\nauthors: A list of the author names.\n\nyear: The year of publication (search for dates near the top or bottom; if not found, estimate based on 'ArXiv' stamps or return 'Unknown').\n\ntopic: A one-phrase classification of the paper (e.g., 'Large Language Models', 'Vector Databases', 'Computer Vision').\n\nOutput only the JSON."""
        },
    }


def config_database():
    return {
        "input_folder_path": "data/raw_pdfs_parsed",
        "chunk_size": 1024,
        "overlap_size": 128 
    }

def config_rag():
    with open("api_keys.json") as f:
        api_keys = json.load(f)
    return {
        "device": "cuda",
        "tavilly_api_key": api_keys["tavilly_web"]
    }