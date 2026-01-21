from __future__ import annotations
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from agents.router import build_llm, router_node
from agents.extractor import build_extractor_llm, extractor_node
from tools.repository import Repo
from tools.calculations import (
    property_summary,
    compare_properties,
    top_expenses,
    pnl,
)


class AppState(TypedDict, total=False):
    messages: Annotated[List[Dict[str, Any]], add_messages]

    intent: str
    confidence: float
    slots: dict

    result: dict
    need_clarification: bool
    clarification_question: str


def build_graph(repo: Repo):
    """
    Construct the LangGraph workflow for the agent.
    
    Args:
        repo: Repository instance for data access.
    Returns:
        Compiled StateGraph.
    """
    router_llm = build_llm()
    extractor_llm = build_extractor_llm()

    def router(state: AppState) -> dict:
        """Node: Route the query using LLM."""
        return router_node(router_llm, state)

    def extractor(state: AppState) -> dict:
        """Node: Extract entities from query using LLM."""
        return extractor_node(extractor_llm, state)

    def calc(state: AppState) -> dict:
        """
        Node: Perform calculations based on intent and slots.
        Validates data, filters datasets, and computes P&L.
        """
        intent = state.get("intent", "fallback")
        slots = state.get("slots", {}) or {}

        props = slots.get("properties", []) or []
        tenants = slots.get("tenants", []) or []
        year = slots.get("year")
        quarter = slots.get("quarter")
        month = slots.get("month")
        top_n = slots.get("top_n", 8)
        breakdown_by = slots.get("breakdown_by") or "ledger_group"

        period = {"year": year, "quarter": quarter, "month": month}
        
        # Get list of valid properties
        valid_properties = repo.list_properties()
        
        # Validate property names and find close matches
        validated_props = []
        invalid_props = []
        for p in props:
            # Try exact match first
            if p in valid_properties:
                validated_props.append(p)
            else:
                # Try fuzzy search
                matches = repo.search_property(p, limit=1)
                if matches and matches[0].lower() == p.lower():
                    validated_props.append(matches[0])
                else:
                    invalid_props.append(p)
        
        # If there are invalid properties, ask for clarification
        if invalid_props:
            return {
                "need_clarification": True,
                "clarification_question": f"The property address '{invalid_props[0]}' does not exist in the dataset. Available properties: {', '.join(valid_properties[:3])}..."
            }
        
        # Replace props with validated ones
        props = validated_props

        # For property_summary and top_expenses - allow portfolio-wide if no property specified
        # But for asset_details we need a specific property
        if intent == "asset_details" and not props:
            return {
                "need_clarification": True,
                "clarification_question": f"Which property would you like details for? For example: {', '.join(valid_properties[:3])}"
            }

        if intent == "compare_properties" and len(props) < 2:
            return {
                "need_clarification": True,
                "clarification_question": f"Please specify at least two properties to compare. For example: {', '.join(valid_properties[:2])}"
            }

        if intent == "list_properties":
            res = {"properties": valid_properties}

        elif intent == "property_summary":
            # If no property specified, show portfolio-wide P&L
            prop = props[0] if props else None
            df = repo.filter_timeframe(year=year, quarter=quarter, month=month)
            if prop:
                df = repo.filter_property(df, prop)
            pnl_data = pnl(df)
            res = {"property": prop or "Entire Portfolio", "period": period, "pnl": pnl_data}

        elif intent == "compare_properties":
            comparison = compare_properties(repo, *props, year=year, quarter=quarter, month=month)
            res = {"comparison": comparison, "period": period}

        elif intent == "top_expenses":
            prop = props[0] if props else None
            expenses = top_expenses(repo, prop, year, quarter, month, breakdown_by, top_n)
            res = {"property": prop, "period": period, "top_expenses": expenses}

        elif intent == "asset_details":
            if not props:
                return {
                    "need_clarification": True,
                    "clarification_question": f"Which property would you like details for? For example: {', '.join(valid_properties[:3])}"
                }
            prop = props[0]
            # Get all data for this property
            df = repo.filter_timeframe(year=year, quarter=quarter, month=month)
            df = repo.filter_property(df, prop)
            pnl_data = pnl(df)
            
            # Get unique values for details
            tenants = df["tenant_name"].dropna().unique().tolist()[:5]
            ledger_groups = df["ledger_group"].dropna().unique().tolist()
            
            res = {
                "property": prop,
                "period": period,
                "pnl": pnl_data,
                "tenants": tenants,
                "ledger_groups": ledger_groups,
                "total_records": len(df)
            }

        elif intent == "tenant_summary":
            if not tenants:
                return {"need_clarification": True, "clarification_question": "Which tenant do you want to analyze?"}
            df = repo.filter_timeframe(year=year, quarter=quarter, month=month)
            df = repo.filter_tenant(df, tenants[0])
            res = {"tenant": tenants[0], "period": period, "pnl": pnl(df)}

        else:
            res = {"note": "fallback"}

        return {"result": res, "need_clarification": False}

    def respond(state: AppState) -> dict:
        """Node: Generate natural language response based on calculation result."""
        intent = state.get("intent", "fallback")
        res = state.get("result", {}) or {}

        if intent == "list_properties":
            text = "Your portfolio contains the following properties: " + ", ".join(res.get("properties", []))

        elif intent == "property_summary":
            pnl_ = res.get("pnl", {})
            net = pnl_.get('net', 0)
            prop_name = res.get('property', 'Entire Portfolio')
            
            # If it's the entire portfolio
            if prop_name == "Entire Portfolio":
                text = f"The total P&L for all your properties is ${net:,.2f}."
            else:
                text = f"The P&L for {prop_name} is ${net:,.2f} (Revenue: ${pnl_.get('revenue', 0):,.2f}, Expenses: ${pnl_.get('expenses', 0):,.2f})."

        elif intent == "compare_properties":
            comparison = res.get("comparison", {})
            properties = comparison.get("properties", [])
            
            # Form response as in example
            parts = []
            for p in properties:
                parts.append(f"the asset at {p['property']} has a Net P&L of ${p['net']:,.2f}")
            
            text = ", while ".join(parts) + "."
            # Capitalize first letter
            text = text[0].upper() + text[1:]

        elif intent == "asset_details":
            prop = res.get('property', 'N/A')
            pnl_ = res.get("pnl", {})
            text = (
                f"Details for the property at {prop}: "
                f"Net P&L - ${pnl_.get('net', 0):,.2f}, "
                f"Total Revenue - ${pnl_.get('revenue', 0):,.2f}, "
                f"Total Expenses - ${pnl_.get('expenses', 0):,.2f}."
            )

        elif intent == "top_expenses":
            items = res.get("top_expenses", [])
            lines = [f"{x.get('ledger_group', 'N/A')}: ${x.get('expense', 0):,.2f}" for x in items]
            text = f"Top expenses for {res.get('property', 'your portfolio')}: " + ", ".join(lines) + "."

        elif intent == "tenant_summary":
            pnl_ = res.get("pnl", {})
            text = f"Tenant {res.get('tenant', 'N/A')} generated a Net P&L of ${pnl_.get('net', 0):,.2f}."

        else:
            text = "I can help you with property details, P&L calculations, comparisons, and expense analysis. Just ask!"

        return {"messages": [{"role": "assistant", "content": text}]}

    def clarify(state: AppState) -> dict:
        """Node: Ask clarification question if needed."""
        q = state.get("clarification_question", "Please clarify your request.")
        return {"messages": [{"role": "assistant", "content": q}]}

    def after_calc(state: AppState) -> str:
        """Edge condition: Determine next step after calculation (respond or clarify)."""
        return "clarify" if state.get("need_clarification") else "respond"

    g = StateGraph(AppState)
    g.add_node("router", router)
    g.add_node("extractor", extractor)
    g.add_node("calc", calc)
    g.add_node("clarify", clarify)
    g.add_node("respond", respond)

    g.add_edge(START, "router")
    g.add_edge("router", "extractor")
    g.add_edge("extractor", "calc")
    g.add_conditional_edges("calc", after_calc)
    g.add_edge("clarify", END)
    g.add_edge("respond", END)

    return g.compile()
