from helper import open_jsonl
from config import config_rag

from pathlib import Path
from tqdm import tqdm
import os
import sys

from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection, AnnSearchRequest, WeightedRanker, MilvusClient
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
    data_folder.mkdir(parents=True, exist_ok=True)
    metadata_path = data_folder / "metadata.jsonl"
    metadata_list = open_jsonl(metadata_path) if metadata_path.exists() else []

    pdf_files = list((data_folder.glob("*.md")))
    pdf_files.sort(key=os.path.getmtime)

    input_data = dict()
    for idx, pdf_file in enumerate(pdf_files):
        metadata = metadata_list[idx] if idx < len(metadata_list) and isinstance(metadata_list[idx], dict) else {}
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
    ef = BGEM3EmbeddingFunction(use_fp16=False, device=config.get("device", "cuda"))
    dense_dim = ef.dim["dense"]
    # Generate embeddings only when chunks are available
    docs_embeddings = ef(chunks_text) if chunks_text else None

    docs = []
    docs_metadata = []
    # prep before insert to database
    for i, doc in enumerate(chunks):
        docs.append(doc.page_content)
        docs_metadata.append(doc.metadata)


    # Connect to Milvus given URI
    connections.connect(uri=config.get("database_path", "./milvus.db"))

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
    col_name = "arag_project"
    if utility.has_collection(col_name):
        Collection(col_name).drop()
    col = Collection(col_name, schema, consistency_level="Bounded")

    # To make vector search efficient, we need to create indices for the vector fields
    sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
    col.create_index("sparse_vector", sparse_index)
    dense_index = {"index_type": "AUTOINDEX", "metric_type": "COSINE"}
    col.create_index("dense_vector", dense_index)
    col.load()

    if not docs:
        print(f"No markdown files found in {path_to_data_folder}. Created empty collection '{col_name}'.")
        return col, ef

    # Transform the sparse array into a list of {index: value} dicts
    formatted_sparse = [
        {int(k): float(v) for k, v in zip(row.indices, row.data)}
        for row in docs_embeddings["sparse"].tocsr() 
    ]

    for i in range(0, len(docs), 50):
        batched_entities = [
            docs[i : i + 50],
            docs_metadata[i : i + 50],
            formatted_sparse[i : i + 50],
            docs_embeddings["dense"][i : i + 50],
        ]
        col.insert(batched_entities)
    
    print("Number of entities inserted:", col.num_entities)
    return col, ef

def hybrid_search(database, embedding_model, query, sparse_weight=1.0, dense_weight=1.0, limit=10):
    query_embeddings = embedding_model([query])
    dense_search_params = {"metric_type": "COSINE", "params": {}}
    dense_req = AnnSearchRequest(
        [query_embeddings["dense"][0]], "dense_vector", dense_search_params, limit=limit
    )
    sparse_search_params = {"metric_type": "IP", "params": {}}
    sparse_req = AnnSearchRequest(
        [query_embeddings["sparse"][[0]]], "sparse_vector", sparse_search_params, limit=limit
    )
    rerank = WeightedRanker(sparse_weight, dense_weight)
    res = database.hybrid_search(
        [sparse_req, dense_req], rerank=rerank, limit=limit, output_fields=["text"]
    )[0]
    docs = [{"text": hit.get("text"), "metadata": hit.get("metadata")} for hit in res]
    return docs

def load_database_and_embedding(database_path, device):
    embedding_model = BGEM3EmbeddingFunction(use_fp16=False, device=device)
    # Connect to Milvus given URI
    connections.connect(uri=database_path)
    if not utility.has_collection("arag_project"):
        raise ValueError(f"Collection 'arag_project' does not exist!")
    
    col = Collection("arag_project")
    col.load()
    print(f"Collection 'arag_project' loaded. Entities: {col.num_entities}")

    return col, embedding_model


def append_parsed_file_to_database(markdown_path, metadata, config, database, embedding_model, batch_size=50):
    """
    Parse a single markdown file into chunks and append embeddings to an existing collection.

    :param markdown_path: Path to the markdown file created by parser.
    :type markdown_path: str
    :param metadata: Metadata dictionary associated with the markdown file.
    :type metadata: dict
    :param config: RAG configuration containing chunking options.
    :type config: dict
    :param database: Loaded Milvus collection.
    :type database: Collection
    :param embedding_model: BGE-M3 embedding model used for dense+sparse vectors.
    :type embedding_model: BGEM3EmbeddingFunction
    :param batch_size: Insert batch size.
    :type batch_size: int
    :return: Number of inserted chunks.
    :rtype: int
    """
    markdown_path = Path(markdown_path)
    if not markdown_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {markdown_path}")

    with open(markdown_path, "r", encoding="utf-8") as f:
        markdown_content = f.read()

    split_markdown_documents = split_data(markdown_content, config)
    if not split_markdown_documents:
        return 0

    for doc in split_markdown_documents:
        if metadata:
            doc.metadata.update(metadata)

    docs = [doc.page_content for doc in split_markdown_documents if doc.page_content]
    docs_metadata = [doc.metadata for doc in split_markdown_documents if doc.page_content]

    if not docs:
        return 0

    docs_embeddings = embedding_model(docs)
    formatted_sparse = [
        {int(k): float(v) for k, v in zip(row.indices, row.data)}
        for row in docs_embeddings["sparse"].tocsr()
    ]

    for i in range(0, len(docs), batch_size):
        batched_entities = [
            docs[i : i + batch_size],
            docs_metadata[i : i + batch_size],
            formatted_sparse[i : i + batch_size],
            docs_embeddings["dense"][i : i + batch_size],
        ]
        database.insert(batched_entities)

    database.flush()
    database.load()
    return len(docs)
    
if __name__ == "__main__":
    config = config_rag()
    data_preprocessing(config)