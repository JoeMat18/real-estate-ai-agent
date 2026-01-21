from __future__ import annotations
import os
from langchain_openai import ChatOpenAI
from .schemas import RouterOut

def build_llm(model: str | None = None):
    """
    Build the LLM client with structured output for routing.
    """
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=0)
    return llm.with_structured_output(RouterOut, method="json_schema")

def router_node(llm, state: dict) -> dict:
    """
    Classify the user intent based on the last message.
    """
    last_msg = state["messages"][-1]
    # LangGraph uses Message objects, not dicts
    user_text = last_msg.content if hasattr(last_msg, 'content') else last_msg["content"]

    prompt = f"""
You are an intent classifier for a real estate financial dataset.
Available intents:
- list_properties: show a list of all properties (e.g. "what properties do you have", "show all buildings")
- property_summary: P&L (profit/loss) for a single property OR the entire portfolio for a period (e.g. "P&L for 2024", "total P&L for all properties", "net profit")
- compare_properties: compare 2+ properties by P&L for a period (explicitly mentioning comparison of 2 or more assets)
- asset_details: detailed information about a single asset (e.g. "tell me about Building X", "details for property")
- top_expenses: top expenses for a property/period
- tenant_summary: P&L by tenant for a period
- fallback: everything else

IMPORTANT: if the user asks about "all properties", "entire portfolio", or "total P&L" -> classify as property_summary (NOT list_properties).

Return intent and confidence (0..1).

Query: {user_text}
""".strip()

    out: RouterOut = llm.invoke(prompt)
    return {"intent": out.intent, "confidence": out.confidence}