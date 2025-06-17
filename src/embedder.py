import os
import sys
import logging
import time
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
from typing import List, Dict
load_dotenv(find_dotenv()) # also search parent dirs for .env
import argparse
import PyPDF2
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from fastembed.common.model_description import PoolingType, ModelSource


load_dotenv()
local_model_path = os.getenv("LOCAL_MODEL_PATH")
if local_model_path:
    os.environ["HF_HOME"] = local_model_path
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(local_model_path, "transformers")
    os.environ["HF_HUB_CACHE"] = os.path.join(local_model_path, "hub")


log_level_str = os.getenv('LOG_LEVEL', 'INFO').upper()
log_level = getattr(logging, log_level_str, logging.INFO)
# get log format from env
log_format = os.getenv('LOG_FORMAT', '%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s')
# Configure logging
logging.basicConfig(level=log_level, format=log_format)

logger = logging.getLogger(__name__)


# Function to read PDF and extract text
def extract_text_from_pdf(path: str) -> str:
    logger.info(f"📄 Opening PDF: {path}")
    text = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            logger.info(f"Page {i+1}: {len(page_text)} characters extracted")
            text.append(page_text)
    return "\n".join(text)

# Semantic chunking into overlapping token windows
def chunk_text(text: str, max_tokens: int = 500, overlap_tokens: int = 50) -> List[str]:
    tokens = text.split()
    chunks = []
    start = 0
    total_tokens = len(tokens)
    while start < total_tokens:
        end = min(start + max_tokens, total_tokens)
        chunk = " ".join(tokens[start:end])
        logger.info(f"📦 Chunk tokens {start}-{end} (length={len(chunk.split())})")
        chunks.append(chunk)
        if end == total_tokens:
            break
        start = end - overlap_tokens
    logger.info(f"✨ Total chunks created: {len(chunks)}")
    return chunks

def store_chunks(
    chunks: List[str],
    collection_name: str,
    client: QdrantClient,
    metadata: List[Dict]
):
    logger.info(f"🚀 Storing {len(chunks)} chunks into collection '{collection_name}'")

    # FastEmbed mixin: embed & upsert in one call
    ids = list(range(1, len(chunks) + 1))
    client.add(
        collection_name=collection_name,
        documents=chunks,
        metadata=metadata,
        ids=ids
    )
    logger.info("✅ Storage complete")


# Core main function accepts PDF path and collection name
def main(pdf_path: str, collection: str):

    if not local_model_path:
        logger.error("🛑 LOCAL_MODEL_PATH is not set in .env")
        sys.exit(1)

    model = os.getenv("DEFAULT_EMBEDDING_MODEL")
    if not model:
        logger.error("🛑 DEFAULT_EMBEDDING_MODEL is not set in .env")
        sys.exit(1)
    logger.info(f"🤖 Using embedding model: {model}")


    text = extract_text_from_pdf(pdf_path)


    start_chunking = time.perf_counter()
    chunks = chunk_text(text)
    end_chunking = time.perf_counter()
    total_chunking = end_chunking - start_chunking
    # Split into whole seconds and leftover milliseconds
    seconds = int(total_chunking)  # whole seconds
    milliseconds = (total_chunking - seconds) * 1000  # fractional ms
    logger.info(f"⏱️ total chunking time: {seconds}s {milliseconds:.3f}ms")

    metadata = [{"source": str(pdf_path), "chunk_index": idx} for idx in range(len(chunks))]

    custom_model_name = "my_model"

    # register a custom model to avid fastembedd warning
    # Warning: The model intfloat/multilingual-e5-large now uses mean pooling instead of CLS embedding. In order to preserve the previous behaviour, consider either pinning fastembed version to 0.5.1 or using `add_custom_model` functionality.
    #  model = TextEmbedding(model_name=model_name, **options)
    TextEmbedding.add_custom_model(
        model=custom_model_name,
        pooling=PoolingType.MEAN,
        normalization=True,
        sources=ModelSource(hf=model),
        dim=1024,
        model_file="model.onnx",  # match repository root
        additional_files=["model.onnx_data"],  # download the external-data file

    )

    client = QdrantClient(
        url = os.getenv("QDRANT_URL", "localhost"),
        port = int(os.getenv("QDRANT_PORT", 6333)),
        prefer_grpc = True
    )

    client.set_model(embedding_model_name = custom_model_name,
        cache_dir = local_model_path)

    existing = [c.name for c in client.get_collections().collections]
    if collection in existing:
        logging.info(
            f"🗑️ deleting existing collection '{collection}' (incompatible vectors)"
        )
        client.delete_collection(collection_name=collection)

    # only needed for performance count
    embedding_model = TextEmbedding(model_name=custom_model_name)
    start_embed = time.perf_counter()
    embeddings = list(embedding_model.embed(chunks))
    end_embed = time.perf_counter()
    total_embed = end_embed - start_embed
    # Split into whole seconds and leftover milliseconds
    seconds = int(total_embed)  # whole seconds
    milliseconds = (total_embed - seconds) * 1000  # fractional ms
    logger.info(f"⏱️ total embedding time using fastembed: {seconds}s {milliseconds:.3f}ms")
    # end only needed for performance count

    start_store_chunks = time.perf_counter()
    store_chunks(chunks, collection, client, metadata)
    end_store_chunks = time.perf_counter()
    total_store_chunks = end_store_chunks - start_store_chunks
    seconds = int(total_store_chunks)  # whole seconds
    milliseconds = (total_store_chunks - seconds) * 1000  # fractional ms
    logger.info(f"⏱️ total time to store and embedd on qdrant via fastembed: {seconds}s {milliseconds:.3f}ms")

    # lists all the available embedding models for fatembedd
    # logger.info(TextEmbedding.list_supported_models())
# CLI wrapper
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Embed PDF text into Qdrant using FastEmbed"
    )

    parser.add_argument(
        "--pdf_path",
        type=Path,
        required=True,
        help="Path to the PDF file to process"
    )
    parser.add_argument(
        "--collection",
        type=str,
        required=True,
        help="Qdrant collection name"
    )
    args = parser.parse_args()
    main(args.pdf_path, args.collection)
