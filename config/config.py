import os

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "your_api_key")
OPENAI_MODEL = "gpt-4o-mini"

SERP_API_KEY = os.environ.get("SERP_API_KEY", "your_api_key")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

PDF_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "Indian-Recipes.pdf")
VECTOR_STORE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vector_store")
