import urllib.request
from rag_app.config import CAT_FACTS_PATH

DATA_URL = (
    "https://huggingface.co/ngxson/demo_simple_rag_py/"
    "resolve/main/cat-facts.txt"
)

def ensure_cat_facts():
    if CAT_FACTS_PATH.exists():
        return

    CAT_FACTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(DATA_URL, CAT_FACTS_PATH)


def load_cat_facts() -> list[str]:
    ensure_cat_facts()

    if not CAT_FACTS_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {CAT_FACTS_PATH}. "
            "Run the dataset download step first."
        )

    with CAT_FACTS_PATH.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(lines)} entries")
    return lines