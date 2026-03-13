import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_community.utilities import SerpAPIWrapper
from langchain_core.tools import tool
from config.config import SERP_API_KEY


@tool
def web_search(query: str) -> str:
    """Search the web for real-time information. Use this ONLY when:
    - The recipe knowledge base did not return a useful answer
    - User asks about current ingredient prices (e.g. 'price of saffron today')
    - User asks about ingredient substitutes or local availability
    - User asks about restaurants, food delivery, or ordering food
    - User asks about nutritional info, food trends, or health topics
    - The dish asked about is not found in the recipe knowledge base
    Never use this for standard Indian recipe queries — always try search_recipes first."""
    try:
        os.environ["SERPAPI_API_KEY"] = SERP_API_KEY
        return SerpAPIWrapper().run(query)
    except Exception as e:
        return f"Web search error: {str(e)}"
