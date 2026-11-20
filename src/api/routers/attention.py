"""Items that need a human decision before the system can (or should) proceed."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.api.services import attention_service

router = APIRouter(prefix="/api/v2/attention", tags=["attention"])


class DismissRequest(BaseModel):
    key: str = Field(min_length=1, max_length=300)
    snooze_minutes: int | None = Field(None, gt=0, le=60 * 24 * 7)  # None = until the situation changes


@router.get("")
def list_items():
    items = attention_service.items()
    return {"items": items, "blocking": sum(1 for i in items if i["severity"] == "blocking")}


@router.post("/dismiss")
def dismiss(req: DismissRequest):
    attention_service.dismiss(req.key, req.snooze_minutes)
    return {"ok": True}
