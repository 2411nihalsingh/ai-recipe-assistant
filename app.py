import streamlit as st
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from models.llm import get_chatgroq_model
from utils.rag_utils import search_recipes
from utils.search_utils import web_search


SYSTEM_PROMPT = """You are Rasa, a warm and knowledgeable Indian cooking assistant.

You have two tools:

1. search_recipes — searches the Indian recipe cookbook (PDF).
   Use this FIRST for any cooking question: recipes, ingredients, steps, spices, methods.

2. web_search — searches the live web.
   Use this ONLY when search_recipes returns nothing useful, OR when the user asks about:
   current ingredient prices, local availability, substitutes, restaurants, nutrition, or food trends.

Always try search_recipes first. Fall back to web_search only if needed.

{mode}"""


@st.cache_resource
def get_llm_with_tools():
    llm = get_chatgroq_model()
    return llm.bind_tools([search_recipes, web_search])


def run_chat(messages, mode):
    try:
        mode_text = (
            "Keep your response SHORT and to the point. Summarize key steps only."
            if mode == "concise"
            else "Give a DETAILED response with full ingredient list, step-by-step instructions, tips, and serving suggestions."
        )

        formatted = [SystemMessage(content=SYSTEM_PROMPT.format(mode=mode_text))]
        for msg in messages:
            if msg["role"] == "user":
                formatted.append(HumanMessage(content=msg["content"]))
            else:
                formatted.append(AIMessage(content=msg["content"]))

        llm_with_tools = get_llm_with_tools()
        response = llm_with_tools.invoke(formatted)

        if response.tool_calls:
            formatted.append(response)
            tool_map = {"search_recipes": search_recipes, "web_search": web_search}
            for tc in response.tool_calls:
                result = tool_map[tc["name"]].invoke(tc["args"])
                formatted.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            return get_chatgroq_model().invoke(formatted).content

        return response.content

    except Exception as e:
        return f"Error: {str(e)}"


def main():
    st.set_page_config(page_title="Rasa", page_icon="🍛", layout="wide")
    st.title("🍛 Rasa — Indian Recipe Assistant")
    st.caption("Ask me anything about Indian cooking!")

    with st.sidebar:
        st.header("Settings")
        mode = st.radio(
            "Response Mode",
            options=["concise", "detailed"],
            format_func=lambda x: "⚡ Concise" if x == "concise" else "Detailed",
        )
        st.divider()
        st.markdown("**When each tool is used**")
        st.markdown(" **RAG** — recipes, ingredients, cooking steps")
        st.markdown(" **Web** — prices, substitutes, restaurants, trends")
        st.divider()
        if st.button(" Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown("Namaste! 🙏 I'm Rasa. Ask me for a recipe, cooking tips, or current ingredient prices!")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Try: 'How to make paneer butter masala?' or 'Price of jeera today'"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Cooking up an answer..."):
                reply = run_chat(st.session_state.messages, mode)
                st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
