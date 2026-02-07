from pathlib import Path
import asyncio
import httpx
import json
import re

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

from llama_cloud import LlamaCloud, AsyncLlamaCloud

from config import config_parse
from helper import clean_json_text

config = config_parse()

async def parse_file(client, file_path):
    file_obj = await client.files.create(file=file_path, purpose="parse")
    
    try:
        result = await client.parsing.parse(
            file_id=file_obj.id,
            tier="agentic",
            version="latest",

            # Control the ou|tput structure and markdown styling
            output_options={
                "markdown": {
                    "tables": {
                        "output_tables_as_markdown": True,
                    },
                },
                # Saving images for later retrieval
                "images_to_save": ["screenshot","embedded"],
            },

            # Options for controlling how we process the document
            processing_options={
                "ignore": {
                    "ignore_diagonal_text": True,
                },
                "cost_optimizer": {
                    "enable": True
                },
            },

            # Parsed content to include in the returned response
            expand=["text", "markdown", "items", "images_content_metadata"],
        )
    except Exception as e:
        with open("fail_logs.txt", "a") as f:
            f.write(f"'parse error' - {e} on {file_path}")
            return
    return result


def load_model(device='cuda'):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map=device
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")


    return model, processor


def run_model(config, model, processor, image, task, device='cuda'):
    if task == "image_caption":
        sys_prompt = config["prompt_imgcap"]["sys"]
        user_prompt = config["prompt_imgcap"]["user"]
    elif task == "metadata_gen":
        sys_prompt = config["prompt_metagen"]["sys"]
        user_prompt = config["prompt_metagen"]["user"]
    else:
        raise ValueError(f"Invalid task: {task}. Expected 'image_caption' or 'metadata_gen'.")
    
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_prompt},
        ]}
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=1024, temperature=0.2, top_p=0.1, do_sample=True)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


def save_markdown(parsed_content, config, model, processor):
    markdown_content = "\n\n".join([page.markdown for page in parsed_content.markdown.pages])
    output_file = Path(config["output_folder"]) / f"{Path(parsed_content.job.name).stem}.md"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # save the parsed markdown
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    
    # process images from the parsed output only figures skip pages
    for figure in parsed_content.images_content_metadata.images:
        if "page" in figure.filename:
            continue
        
        try:
            generated_cap_text = run_model(config, model, processor, figure.presigned_url, task="image_caption", device=config["device"])
        except Exception as e:
            with open("fail_logs.txt", "a") as f:
                f.write(f"'image caption error' - {e} on {parsed_content.job.name}")
                return
        generated_cap_dict = json.loads(clean_json_text(generated_cap_text))
        converted_texts = [f"{key}: {value}" for key, value in generated_cap_dict.items()]
        caption = "\n\n".join(converted_texts)

        with open(output_file, "a", encoding="utf-8") as f:
            f.write("\n\n\n\n")
            f.write(caption)
    # generate metadata from page 1
    first_page_image = parsed_content.images_content_metadata.images[0]

    try:
        generated_metadata_text = run_model(config, model, processor, first_page_image.presigned_url, task="metadata_gen", device=config["device"])
    except Exception as e:
        with open("fail_logs.txt", "a") as f:
            f.write(f"'file metadata gen error' - {e} on {parsed_content.job.name}")
            return
    metadata = json.loads(clean_json_text(generated_metadata_text))

    metadata_path = Path(config["output_folder"]) / "metadata.jsonl"
    with open(metadata_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(metadata, ensure_ascii=False) + "\n")
    

async def parse_folder(config):
    client = AsyncLlamaCloud(api_key=config["api_key"])
    
    file_folder = Path(config['input_folder'])
    pdf_files = list(file_folder.glob("*.pdf"))

    tasks = [parse_file(client, pdf_file) for pdf_file in pdf_files]

    results = await asyncio.gather(*tasks)
    results = [r for r in results if r is not None]

    # load model
    model, processor = load_model(config["device"])

    # write markdown
    for result in results:
        save_markdown(result, config, model, processor)
    
    return results
        

if __name__ == '__main__':
    asyncio.run(parse_folder(config))