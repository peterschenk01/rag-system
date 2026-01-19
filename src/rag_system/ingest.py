import logging
import urllib.request

from rag_system.config import DATA_PATH, DATA_URL

logger = logging.getLogger(__name__)


def ensure_data_exists() -> None:
    if DATA_PATH.exists():
        logger.info(f"Dataset already exists at: {DATA_PATH}")
        return

    logger.info(f"Dataset not found. Downloading from: {DATA_URL}")

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)


def chunk_dataset(dataset: list[str]) -> list[str]:
    """Chunking strategy: one line per chunk.

    This is intentionally simple and acts as a single place to evolve chunking later.
    """
    chunks = dataset  # one line = one chunk
    return chunks


def load_dataset() -> list[str]:
    logger.info("Loading dataset...")
    ensure_data_exists()

    with DATA_PATH.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    chunks = chunk_dataset(lines)

    logger.info("Dataset loaded.")
    return chunks
