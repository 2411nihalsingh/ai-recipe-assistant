import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_openai import ChatOpenAI
from config.config import OPENAI_API_KEY, OPENAI_MODEL


def get_chatopenai_model():
    try:
        return ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=OPENAI_MODEL,
            temperature=0.3,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize OpenAI model: {str(e)}")