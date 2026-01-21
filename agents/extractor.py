from __future__ import annotations
import os
from langchain_openai import ChatOpenAI
from .schemas import SlotOut

def build_extractor_llm(model: str | None = None):
    """
    Build the LLM client with structured output for entity extraction.
    """
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=0)
    return llm.with_structured_output(SlotOut, method="json_schema")
 
def extractor_node(llm, state: dict) -> dict:
    """
    Extract entities (slots) from the user's query like properties, dates, and tenants.
    """
    last_msg = state["messages"][-1]
    # LangGraph uses Message objects, not dicts
    user_text = last_msg.content if hasattr(last_msg, 'content') else last_msg["content"]

    prompt = f"""
Extract parameters from the user's query.
We have timeframe fields:
- year: 2024 or 2025
- quarter: format "2024-Q4"
- month: format "2025-M02"
We also have property names like "Building 180" and tenants like "Tenant 8".

Rules:
- properties: list of property_name (if 2 properties — list of two)
- tenants: list of tenant_name
- if timeframe is not specified — keep year/quarter/month = null

Query: {user_text}
""".strip()

    out: SlotOut = llm.invoke(prompt)
    return {"slots": out.model_dump()}