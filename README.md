# Rasa — Indian Recipe Assistant

A RAG + Live Web Search chatbot built with LangChain, Groq, and Streamlit.

## Quick Start

1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Add your API keys in `config/config.py`

3. Run
```bash
streamlit run app.py
```

## Features
- RAG over Indian Recipes PDF (FAISS vector store)
- Live web search via SerpAPI for prices, substitutes, trending recipes
- LangChain tool-calling agent (decides RAG vs web automatically)
- Concise / Detailed response modes
- Conversation memory within session
