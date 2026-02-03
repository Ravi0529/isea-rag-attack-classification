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
    action1 = arr[0] if len(arr) > 0 else None
    action2 = arr[1] if len(arr) > 1 else None
    ts = arr[2] if len(arr) > 2 else None
    src_ip = arr[3] if len(arr) > 3 else None
    src_port = _to_int(arr[4] if len(arr) > 4 else None)
    ua = arr[5] if len(arr) > 5 else None
    lang = arr[6] if len(arr) > 6 else None
    extra = arr[7] if len(arr) > 7 else None

    # basic sanity on timestamp
    if not isinstance(ts, str) or not _TS_RE.match(ts):
        return None

    return {
        "ts": ts.replace(" ", "T") + "Z",  # ISO-like (Phase 1)
        "src_ip": src_ip if isinstance(src_ip, str) else None,
        "src_port": src_port,
        "user_agent": ua if isinstance(ua, str) else None,
        "lang": lang if isinstance(lang, str) else None,
        "action1": action1 if isinstance(action1, str) else None,
        "action2": action2 if isinstance(action2, str) else None,
        "extra": extra if isinstance(extra, str) else None,
        "raw_line": s,
    }


def iter_events(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "rb") as f:
        for bline in f:
            line = bline.decode("utf-8", errors="ignore")
            ev = parse_line_to_event(line)
            if ev is not None:
                yield ev
