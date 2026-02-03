from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional


class NormalizedEvent(BaseModel):
    ts: str  # keep ISO string for Phase 1
    src_ip: Optional[str] = None
    src_port: Optional[int] = None
    user_agent: Optional[str] = None
    lang: Optional[str] = None

    # These two appear in your sample as “cmd/action” fields sometimes
    action1: Optional[str] = None
    action2: Optional[str] = None

    # extra field in your sample (sometimes another IP)
    extra: Optional[str] = None

    raw_line: str = Field(..., description="Original raw log line")
