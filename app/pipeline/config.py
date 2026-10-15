import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    firecrawl_api_url: str = os.environ.get("FIRECRAWL_API_URL", "http://localhost:3002")
    ollama_base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    llm_model: str = os.environ.get("LLM_MODEL", "qwen2.5:14b-instruct-q4_K_M")
    embed_model: str = os.environ.get("EMBED_MODEL", "bge-large")
    llm_seed: int = int(os.environ.get("LLM_SEED", "1234"))
    min_words: int = int(os.environ.get("MIN_WORDS", "150"))
    scrape_timeout_ms: int = int(os.environ.get("SCRAPE_TIMEOUT_MS", "30000"))
    map_limit: int = int(os.environ.get("MAP_LIMIT", "5000"))
    # comma-separated substrings: child sitemaps whose URL contains one are
    # skipped entirely (BigCommerce ?type=products, *-product-sitemap.xml, …)
    sitemap_skip_patterns: str = os.environ.get(
        "SITEMAP_SKIP_PATTERNS",
        "type=products,product-sitemap,products-sitemap")
    # optional regex applied to every discovered URL path during triage —
    # rejects e.g. /products/ or /product/ on platforms with no type signal
    exclude_path_regex: str = os.environ.get("EXCLUDE_PATH_REGEX", "")
    data_dir: str = os.environ.get("DATA_DIR", "./data")
    checkpoint_db: str = os.environ.get("CHECKPOINT_DB", "./checkpoints/corpus.db")
    mongo_uri: str = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
    mongo_db: str = os.environ.get("MONGO_DB", "extractor")

    # Deterministic Ollama inference options (applied to every chat call)
    llm_options: dict = None

    def __post_init__(self):
        object.__setattr__(self, "llm_options", {
            "temperature": 0,
            "seed": self.llm_seed,
            "top_k": 1,
            "num_ctx": 8192,
            "num_predict": 200,   # verdict is ~20 tokens; caps runaway generation
        })


SETTINGS = Settings()
