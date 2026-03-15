import os

GROQ_API_KEY = "YOUR_API_KEY"
SERP_API_KEY = "YOUR_API_KEY"

GROQ_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

PDF_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "Indian-Recipes.pdf")
VECTOR_STORE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vector_store")
