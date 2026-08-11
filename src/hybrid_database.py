import hashlib
import json
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pymilvus import (
    AnnSearchRequest,
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    WeightedRanker,
    connections,
    utility,
)
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from .configuration import config_rag
from .helper import open_jsonl

# Bumped from "arag_project" alongside the auto_id=False schema migration
# (PROJECT_SPEC.md §5.2) -- old and new indexes can coexist while verifying.
COLLECTION_NAME = "arag_project_v2"

# Descriptive fields from metadata.jsonl worth carrying onto every chunk for
# citation/attribution. Bookkeeping fields (schema_version, content_sha256,
# n_chunks, parsed_at, parser, vlm_captioner, verified_by) describe the
# document, not the chunk, and stay out of per-chunk metadata.
CHUNK_METADATA_FIELDS = ("title", "authors", "year", "topic", "arxiv_id")


def chunk_content_sha8(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


# pip install "transformers<5.0.0" "FlagEmbedding>=1.2.0" --upgrade

# data construction to embed into database
def prepare_input_data(data_folder_path: str, metadata_path: str) -> dict:
    """
    prepare input_data in python dict having pdf file path to its corresponding metadata produced
    
    :param data_folder_path: path of the folder that needs to be input for database
    :type data_folder_path: str
    :return: Python dictionary that has markdown file paths to its corresponding metadata
    :rtype: dict
    """
    data_folder = Path(data_folder_path)
    data_folder.mkdir(parents=True, exist_ok=True)

    metadata_file = Path(metadata_path)
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"metadata.jsonl not found at {metadata_file}. Run deploy/build_metadata.py first."
        )
    metadata_records = open_jsonl(metadata_file)

    md_files_by_name = {p.name: p for p in data_folder.glob("*.md")}

    input_data = {}
    claimed_md_files = set()
    for record in metadata_records:
        doc_id = record.get("doc_id")
        source_md = record.get("source_md")
        if not doc_id or not source_md:
            raise ValueError(f"metadata.jsonl row missing doc_id/source_md: {record}")
        if doc_id in input_data:
            raise ValueError(f"Duplicate doc_id in metadata.jsonl: {doc_id}")
        if source_md not in md_files_by_name:
            raise ValueError(f"doc_id {doc_id!r}: source_md {source_md!r} not found under {data_folder}")
        input_data[doc_id] = {"path": md_files_by_name[source_md], "metadata": record}
        claimed_md_files.add(source_md)

    orphan_md_files = set(md_files_by_name) - claimed_md_files
    if orphan_md_files:
        raise ValueError(f".md files under {data_folder} with no metadata.jsonl row: {sorted(orphan_md_files)}")

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


def build_chunks(config: dict) -> list:
    """Chunk every document and assign deterministic chunk_ids. A pure
    function of (input_folder_path, metadata_path, chunk_size,
    overlap_size) with no embeddings and no Milvus -- separated out so
    chunk-ID determinism is testable without a real (slow) index build.
    """
    path_to_data_folder = config.get("input_folder_path", "artifacts/parsed_md")
    metadata_path = config.get("metadata_path", "artifacts/metadata.jsonl")
    input_data = prepare_input_data(path_to_data_folder, metadata_path)

    # list for storing data chunks; iterate doc_ids in sorted order so
    # chunk_index (and therefore every chunk_id) is independent of dict
    # iteration order
    chunks = []

    for doc_id in sorted(input_data.keys()):
        entry = input_data[doc_id]
        input_path = entry["path"]
        metadata = entry["metadata"]
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

    docs_ids = []
    docs = []
    docs_metadata = []
    # prep before insert to database
    for doc in chunks:
        docs_ids.append(doc.metadata["chunk_id"])
        docs.append(doc.page_content)
        docs_metadata.append(doc.metadata)

    # Connect to Milvus given URI
    connections.connect(uri=config.get("database_path", "./milvus.db"))

    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.VARCHAR,
            description="deterministic chunk ID: {doc_id}::{chunk_index:04d}::{content_sha8}",
            is_primary=True,
            auto_id=False,
            max_length=256,
        ),
        FieldSchema(
            name="text",
            dtype=DataType.VARCHAR,
            description="content of chunk",
            max_length=65535,
        ),
        FieldSchema(
            name="metadata",
            dtype=DataType.JSON,
            description="metadata of chunk",
        ),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
    ]
    schema = CollectionSchema(fields)

    # Create collection
    if utility.has_collection(COLLECTION_NAME):
        Collection(COLLECTION_NAME).drop()
    col = Collection(COLLECTION_NAME, schema, consistency_level="Bounded")

    # To make vector search efficient, we need to create indices for the vector fields
    sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
    col.create_index("sparse_vector", sparse_index)
    dense_index = {"index_type": "AUTOINDEX", "metric_type": "COSINE"}
    col.create_index("dense_vector", dense_index)
    col.load()

    if not docs:
        print(f"No markdown files found in {path_to_data_folder}. Created empty collection '{COLLECTION_NAME}'.")
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

    _write_back_chunk_counts(metadata_path, n_chunks_by_doc)
    print(f"Wrote back n_chunks for {len(n_chunks_by_doc)} documents to {metadata_path}")

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
    print(f"Collection '{COLLECTION_NAME}' loaded. Entities: {col.num_entities}")

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

    docs_ids = []
    docs = []
    docs_metadata = []
    for chunk_index, doc in enumerate(split_markdown_documents):
        if not doc.page_content:
            continue
        content_sha8 = chunk_content_sha8(doc.page_content)
        # Not a corpus doc_id (uploads aren't in metadata.jsonl / the
        # frozen corpus) -- namespaced separately so it can never collide
        # with a real chunk_id.
        docs_ids.append(f"upload::{markdown_path.stem}::{chunk_index:04d}::{content_sha8}")
        docs.append(doc.page_content)
        docs_metadata.append(doc.metadata)

    if not docs:
        return 0

    docs_embeddings = embedding_model(docs)
    formatted_sparse = [
        {int(k): float(v) for k, v in zip(row.indices, row.data)}
        for row in docs_embeddings["sparse"].tocsr()
    ]

    for i in range(0, len(docs), batch_size):
        batched_entities = [
            docs_ids[i : i + batch_size],
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
