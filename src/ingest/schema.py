from __future__ import annotations
from typing import Optional

from pydantic import BaseModel, Field


class NormalizedEvent(BaseModel):
    captured_cmd: Optional[str] = None
    captured_args: Optional[str] = None
    timestamp: Optional[str] = None
    source_ip: Optional[str] = None
    source_port: Optional[int] = None
    user_agent: Optional[str] = None
    language: Optional[str] = None
    x_forwarded_for_real_ip: Optional[str] = Field(
        default=None, alias="x_forwarded_for/real-ip"
    )
