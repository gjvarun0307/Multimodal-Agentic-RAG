from helper import open_jsonl
from config import config_database

from pathlib import Path
from tqdm import tqdm
import os
import sys

from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_milvus import Milvus
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

# pip install "transformers<5.0.0" "FlagEmbedding>=1.2.0" --upgrade

# data construction to embed into database
def prepare_input_data(data_folder_path: str) -> dict:
    """
    prepare input_data in python dict having pdf file path to its corresponding metadata produced
    
    :param data_folder_path: path of the folder that needs to be input for database
    :type data_folder_path: str
    :return: Python dictionary that has markdown file paths to its corresponding metadata
    :rtype: dict
    """
    # prepare the input data path to metadata
    data_folder = Path(data_folder_path)
    metadata_list = open_jsonl(data_folder / "metadata.jsonl")

    pdf_files = list((data_folder.glob("*.md")))
    pdf_files.sort(key=os.path.getmtime)

    input_data = dict()
    for pdf_file, metadata in zip(pdf_files, metadata_list):
        input_data[pdf_file] = metadata

    return input_data

def split_data(markdown_document: str, config: dict):
    # never seen a paper use ###### in their sections
    headers_to_split_on = [
        ("#", "Title"),
        ("##", "Section"),
        ("###", "Subsection"),
        ("######", "Caption"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=True
    )
    md_header_splits = markdown_splitter.split_text(markdown_document)
    # return md_header_splits

    # Char-level splits
    chunk_size = config["chunk_size"]
    chunk_overlap = config["overlap_size"]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ". ", " ", ""]
    )

    # Split
    splits = text_splitter.split_documents(md_header_splits)
    return splits

def data_preprocessing(config: dict):
    # get the input data
    path_to_data_folder = config["input_folder_path"]
    input_data = prepare_input_data(path_to_data_folder)

    # list for storing data chunks
    chunks = []

    for input_path, metadata in input_data.items():
        with open(input_path, "r") as f:
            markdown_content = f.read()
        split_markdown_documents = split_data(markdown_content, config)
        for doc in split_markdown_documents:
            doc.metadata.update(metadata)
        chunks.extend(split_markdown_documents)
    
    chunks_text = []
    # now that we have chunks of data we input it into database in batches
    for chunk in chunks:
        chunks_text.append(chunk.page_content)
    
    # get embedding model(BGE-M3)
    ef = BGEM3EmbeddingFunction(use_fp16=False, device="cuda")
    dense_dim = ef.dim["dense"]
    # Generate embeddings using BGE-M3 model
    docs_embeddings = ef(chunks_text)

    data_rows = []
    # prep before insert to database
    for i, doc in enumerate(chunks):
        entity = {
            "text": doc.page_content,
            "dense_vector": docs_embeddings['dense'][i],
            "sparse_vector": docs_embeddings['sparse'][i],
            "metadata": doc.metadata,
        }
        data_rows.append(entity)

    # Connect to Milvus given URI
    connections.connect(uri="./milvus.db")

    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            description="primary key",
            is_primary=True,
            auto_id=True,
            max_length=100
        ),
        FieldSchema(
            name="text",
            dtype=DataType.VARCHAR,
            description="content of chunk",
            max_length=65535,
        ),FieldSchema(
            name="metadata",
            dtype=DataType.JSON,
            description="metadata of chunk",
        ),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
    ]
    schema = CollectionSchema(fields)

    # Create collection
    col_name = "arag_test"
    if utility.has_collection(col_name):
        Collection(col_name).drop()
    col = Collection(col_name, schema, consistency_level="Bounded")

    # To make vector search efficient, we need to create indices for the vector fields
    sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
    col.create_index("sparse_vector", sparse_index)
    dense_index = {"index_type": "AUTOINDEX", "metric_type": "COSINE"}
    col.create_index("dense_vector", dense_index)
    col.load()

    batch_size = 50
    for i in tqdm(range(0, len(data_rows), batch_size), desc="Inserting batches into database"):
        batch = data_rows[i : i + batch_size]
        try:
            col.insert(batch)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(3)
    
    print("Number of entities inserted:", col.num_entities)

    query = input("Enter your search query: ")
    print(query)

    # Generate embeddings for the query
    query_embeddings = ef([query])
    


    
if __name__ == "__main__":
    config = config_database()
    data_preprocessing(config)