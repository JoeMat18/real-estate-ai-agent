from __future__ import annotations
from typing import Optional
import pandas as pd
from .repository import Repo

def pnl(df: pd.DataFrame) -> dict:
    """Calculate revenue, expenses, and net profit from a dataframe."""
    revenue = df[df["ledger_type"] == "revenue"]["profit"].sum()
    expenses = df[df["ledger_type"] == "expenses"]["profit"].sum()
    net = df["profit"].sum()
    return {"revenue": float(revenue), "expenses": float(expenses), "net": float(net)}

def property_summary(
    repo: Repo,
    property_name: Optional[str],
    year: Optional[int],
    quarter: Optional[str],
    month: Optional[str]
) -> dict:
    """Calculate P&L for a specific property (or all) over a timeframe."""
    df = repo.filter_timeframe(year=year, quarter=quarter, month=month)
    df = repo.filter_property(df, property_name)
    return pnl(df)

def compare_properties(
    repo: Repo,
    *property_names: str,
    year: Optional[int] = None,
    quarter: Optional[str] = None,
    month: Optional[str] = None
) -> dict:
    """
    Compare P&L for multiple properties (2 or more).
    
    Usage:
        compare_properties(repo, "Property A", "Property B")
        compare_properties(repo, "Property A", "Property B", "Property C", year=2024)
    """
    if len(property_names) < 2:
        raise ValueError("At least 2 properties required for comparison")
    
    results = []
    for prop_name in property_names:
        summary = property_summary(repo, prop_name, year, quarter, month)
        results.append({"property": prop_name, **summary})
    
    return {"properties": results, "count": len(results)}

def top_expenses(
    repo: Repo,
    property_name: Optional[str],
    year: Optional[int],
    quarter: Optional[str],
    month: Optional[str],
    group_by: str = "ledger_group",
    top_n: int = 8
) -> list[dict]:
    """Get top expense categories for a property over a timeframe."""
    df = repo.filter_timeframe(year=year, quarter=quarter, month=month)
    df = repo.filter_property(df, property_name)
    df = df[df["ledger_type"] == "expenses"]

    if group_by not in df.columns:
        return []

    g = df.groupby(group_by, dropna=False)["profit"].sum().sort_values()
    out = []
    for k, v in g.head(top_n).items():
        out.append({group_by: str(k), "expense": float(v)})
    return out