from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Iterable
import pandas as pd
import re 


MONTH_RE = re.compile(r"^(?P<y>\d{4})-M(?P<m>\d{2})$")
QUARTER_RE = re.compile(r"^(?P<y>\d{4})-Q(?P<q>[1-4])$")

def _norm(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = str(s).strip()
    return s if s else None

@dataclass
class Repo:
    df: pd.DataFrame

    @classmethod
    def from_parquet(cls, path: str) -> "Repo":
        """
        Load dataset from a parquet file, normalize columns, and create a Repo instance.
        """
        df = pd.read_parquet(path)
        # normalization
        for c in ["entity_name","property_name","tenant_name","ledger_type",
                  "ledger_group","ledger_category","ledger_description","month","quarter","year"]:
            if c in df.columns:
                df[c] = df[c].map(_norm)
        df["profit"] = pd.to_numeric(df["profit"], errors="coerce").fillna(0.0)
        # year like int
        df["year_int"] = pd.to_numeric(df["year"], errors="coerce")
        return cls(df=df)
    
    def list_properties(self) -> list[str]:
        """List all unique existing property names."""
        props = (
            self.df["property_name"]
            .dropna()
            .drop_duplicates()
            .sort_values()
            .to_list()
        )
        return props

    def search_property(self, query: str, limit: int = 5) -> list[str]:
        """Search for property names containing the query substring."""
        q = query.lower().strip()
        props = self.list_properties()
        hits = [p for p in props if q in p.lower()]
        return hits[:limit] if hits else props[:limit]

    def filter_timeframe(
        self,
        *,
        year: Optional[int] = None,
        quarter: Optional[str] = None,
        month: Optional[str] = None,
    ) -> pd.DataFrame:
        """Filter the dataframe based on year, quarter, or month."""
        df = self.df
        if year is not None:
            df = df[df["year_int"] == year]
        if quarter is not None:
            df = df[df["quarter"] == quarter]
        if month is not None:
            df = df[df["month"] == month]
        return df

    def filter_property(self, df: pd.DataFrame, property_name: Optional[str]) -> pd.DataFrame:
        """Filter dataframe by property name."""
        if property_name is None:
            return df
        return df[df["property_name"].fillna("") == property_name]

    def filter_tenant(self, df: pd.DataFrame, tenant_name: Optional[str]) -> pd.DataFrame:
        """Filter dataframe by tenant name."""
        if tenant_name is None:
            return df
        return df[df["tenant_name"].fillna("") == tenant_name]


    def property_pnl(
        self,
        *,
        property_name: Optional[str],
        year: Optional[int] = None,
        quarter: Optional[str] = None,
        month: Optional[str] = None,
    ) -> dict:
        df = self.filter_timeframe(year=year, quarter=quarter, month=month)
        df = df[df["property_name"].fillna("") == (property_name or "")]
        # revenue/expenses/net
        revenue = df[df["ledger_type"] == "revenue"]["profit"].sum()
        expenses = df[df["ledger_type"] == "expenses"]["profit"].sum()
        net = df["profit"].sum()
        return {"revenue": float(revenue), "expenses": float(expenses), "net": float(net)}

    def breakdown(
        self,
        *,
        property_name: Optional[str],
        group_by: str = "ledger_group",
        year: Optional[int] = None,
        quarter: Optional[str] = None,
        month: Optional[str] = None,
        top_n: int = 8,
    ) -> list[dict]:
        df = self.filter_timeframe(year=year, quarter=quarter, month=month)
        df = df[df["property_name"].fillna("") == (property_name or "")]
        if group_by not in df.columns:
            return []
        g = (
            df.groupby(group_by, dropna=False)["profit"]
            .sum()
            .sort_values()
        )
        # сортируем: самые большие расходы (самые отрицательные) сверху
        out = []
        for k, v in g.head(top_n).items():
            out.append({group_by: str(k), "profit": float(v)})
        return out