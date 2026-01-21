import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from tools.repository import Repo
from graph.app_graph import build_graph

load_dotenv()

st.set_page_config(page_title="Real Estate Agent", layout="centered")
st.title("Real Estate Agent")

if not os.getenv("OPENAI_API_KEY"):
    st.warning("Поставь OPENAI_API_KEY в .env или переменные окружения.")
    st.stop()


def get_role(msg):
    """Get role from message (dict or LangChain Message object)."""
    if hasattr(msg, 'type'):
        return "assistant" if msg.type == "ai" else "user"
    return msg.get("role", "user")


def get_content(msg):
    """Get content from message (dict or LangChain Message object)."""
    if hasattr(msg, 'content'):
        return msg.content
    return msg.get("content", "")


@st.cache_resource
def load_app():
    """
    Initialize the application graph and repository.
    Cached by Streamlit to avoid reloading on every rerun.
    """
    repo = Repo.from_parquet("data/cortex.parquet")
    app = build_graph(repo)
    return repo, app


repo, app = load_app()

# Initialize with a greeting message
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        AIMessage(content="Hello! I can help you analyze your real estate portfolio. Ask me something like: 'P&L for Building 180 in 2024' or 'Show all properties'")
    ]

# Display chat history
for m in st.session_state["messages"]:
    role = get_role(m)
    content = get_content(m)
    with st.chat_message(role):
        st.write(content)

# Handle user input
user_input = st.chat_input("Type your request here...")
if user_input:
    # Add user message
    st.session_state["messages"].append(HumanMessage(content=user_input))
    
    with st.chat_message("user"):
        st.write(user_input)

    # Invoke LangGraph
    with st.spinner("Thinking..."):
        state = {"messages": st.session_state["messages"]}
        out = app.invoke(state)
        
        # Update messages with LangGraph output
        st.session_state["messages"] = out["messages"]

    # Display the last assistant message
    last_msg = st.session_state["messages"][-1]
    with st.chat_message("assistant"):
        st.write(get_content(last_msg))
