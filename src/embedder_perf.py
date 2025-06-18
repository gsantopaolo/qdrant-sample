import os
import sys
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# 1️⃣ Load env and set caches BEFORE imports
load_dotenv(find_dotenv())
local_model_path = os.getenv("LOCAL_MODEL_PATH")
CACHE_ROOT = Path(os.getenv(local_model_path, "../data/models")).resolve()
os.environ["HF_HOME"] = str(CACHE_ROOT)
os.environ["HF_HUB_CACHE"] = str(CACHE_ROOT)
# os.environ["TRANSFORMERS_CACHE"] = str(CACHE_ROOT)
os.environ["FASTEMBED_CACHE"] = str(CACHE_ROOT)   # optional
os.environ["FASTEMBED_CACHE_PATH"] = str(CACHE_ROOT)
qdrant_host = os.getenv("QDRANT_HOST", "localhost")
qdrant_port = int(os.getenv("QDRANT_PORT", 6333))

from utils import *

import  logging, argparse
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from fastembed.common.model_description import PoolingType, ModelSource


log_level_str = os.getenv('LOG_LEVEL', 'INFO').upper()
log_level = getattr(logging, log_level_str, logging.INFO)
# get log format from env
log_format = os.getenv('LOG_FORMAT', '%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s')
# Configure logging
logging.basicConfig(level=log_level, format=log_format)
logger = logging.getLogger(__name__)



# Core main function accepts PDF path and collection name
def main(pdf_path: str):

    if not local_model_path:
        logger.error("🛑 LOCAL_MODEL_PATH is not set in .env")
        sys.exit(1)

    model = os.getenv("QDRANT_EMBEDDING_MODEL")
    if not model:
        logger.error("🛑 QDRANT_EMBEDDING_MODEL is not set in .env")
        sys.exit(1)
    logger.info(f"🤖 using qdrant embedding model: {model}")

    hf_model = os.getenv("HF_EMBEDDING_MODEL")
    if not hf_model:
        logger.error("🛑 HF_EMBEDDING_MODEL is not set in .env")
        sys.exit(1)
    logger.info(f"🤖 using HF embedding model: {hf_model}")

    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)

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

    # need to initialize a QdrantClient only to set the cache dir
    client = QdrantClient(url=qdrant_host, port=qdrant_port, prefer_grpc=True)
    client.set_model(embedding_model_name="my_model", cache_dir=str(CACHE_ROOT))

    logger.info("️\n")
    logger.info("️👉 start embedding using fastembed")
    embedding_model = TextEmbedding(model_name=custom_model_name)
    embeddings = list(embedding_model.embed(chunks))

    logger.info("️\n")
    logger.info("️👉 start embedding using HF SentenceTransformers")
    embeddings = embed_with_hf(chunks)

    # lists all the available embedding models for fatembedd
    # logger.info(TextEmbedding.list_supported_models())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Embedding comparison fastembed vs HF"
    )

    parser.add_argument(
        "--pdf_path",
        type=Path,
        required=True,
        help="Path to the PDF file to process"
    )

    args = parser.parse_args()
    main(args.pdf_path)
