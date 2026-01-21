from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

Intent = Literal[
    "list_properties",
    "property_summary",
    "compare_properties",
    "asset_details",
    "top_expenses",
    "tenant_summary",
    "fallback",
]

class RouterOut(BaseModel):
    intent: Intent
    confidence: float = Field(ge=0, le=1)

class SlotOut(BaseModel):
    properties: List[str] = Field(default_factory=list)
    tenants: List[str] = Field(default_factory=list)

    year: Optional[int] = None
    quarter: Optional[str] = None
    month: Optional[str] = None

    breakdown_by: Optional[Literal["ledger_group", "ledger_category", "tenant_name"]] = None
    top_n: Optional[int] = Field(default=8, ge=1, le=20)