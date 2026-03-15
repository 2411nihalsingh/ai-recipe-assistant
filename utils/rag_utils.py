import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from models.embeddings import get_embedding_model
from config.config import PDF_PATH, VECTOR_STORE_PATH


@st.cache_resource(show_spinner="Building recipe knowledge base...")
def get_vector_store():
    try:
        embeddings = get_embedding_model()

        if os.path.exists(VECTOR_STORE_PATH):
            return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)

        loader = PyPDFLoader(PDF_PATH)
        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(loader.load())
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(VECTOR_STORE_PATH)
        return vector_store
    except Exception as e:
        raise RuntimeError(f"Failed to build vector store: {str(e)}")


@tool
def search_recipes(query: str) -> str:
    """Search the Indian recipe cookbook for recipes, ingredients, cooking steps,
    spice combinations, and preparation tips. Use this for any question about how
    to cook a dish, what ingredients are needed, or traditional cooking methods."""
    try:
        results = get_vector_store().similarity_search(query, k=5)
        if not results:
            return "No relevant recipe found in the knowledge base."
        return "\n\n---\n\n".join([doc.page_content for doc in results])
    except Exception as e:
        return f"Knowledge base error: {str(e)}"
