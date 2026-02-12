from __future__ import annotations
import re
import orjson
from typing import Any, Dict, Iterator, Optional

_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")


def _to_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _to_nullable_str(x: Any) -> Optional[str]:
    if not isinstance(x, str):
        return None
    s = x.strip()
    if not s:
        return None
    if s.lower() in {"null", "none", "nan", "na"}:
        return None
    return s


def parse_line_to_event(line: str) -> Optional[Dict[str, Any]]:
    """
    Your lines look like JSON arrays, e.g. but the actual log file is in .log format
    [null,null,"2023-01-08 08:07:15","104.28.209.153","61901","Mozilla/5.0 ...","en","104.28.209.153"]
    or
    ["previewFilePath","/etc/passwd","2024-01-30 07:57:26","152.58.34.14","64940","Mozilla/5.0 ...",null,null]
    """
    s = line.strip()
    if not s:
        return None

    try:
        arr = orjson.loads(s)
    except Exception:
        return None

    if not isinstance(arr, list) or len(arr) < 6:
        return None

    # Expected positions based on your sample
    captured_cmd = _to_nullable_str(arr[0] if len(arr) > 0 else None)
    captured_args = _to_nullable_str(arr[1] if len(arr) > 1 else None)
    timestamp = _to_nullable_str(arr[2] if len(arr) > 2 else None)
    source_ip = _to_nullable_str(arr[3] if len(arr) > 3 else None)
    source_port = _to_int(arr[4] if len(arr) > 4 else None)
    user_agent = _to_nullable_str(arr[5] if len(arr) > 5 else None)
    language = _to_nullable_str(arr[6] if len(arr) > 6 else None)
    x_forwarded_for_real_ip = _to_nullable_str(arr[7] if len(arr) > 7 else None)

    # Keep only parseable timestamp format and normalize to ISO-like string.
    if timestamp is not None:
        if _TS_RE.match(timestamp):
            timestamp = timestamp.replace(" ", "T") + "Z"
        else:
            timestamp = None

    return {
        "captured_cmd": captured_cmd,
        "captured_args": captured_args,
        "timestamp": timestamp,
        "source_ip": source_ip,
        "source_port": source_port,
        "user_agent": user_agent,
        "language": language,
        "x_forwarded_for/real-ip": x_forwarded_for_real_ip,
    }


def iter_events(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "rb") as f:
        for bline in f:
            line = bline.decode("utf-8", errors="ignore")
            ev = parse_line_to_event(line)
            if ev is not None:
                yield ev
