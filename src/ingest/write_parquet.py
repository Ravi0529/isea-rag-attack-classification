from __future__ import annotations

import os
from typing import Any, Dict, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Fixed ingest schema aligned with raw log array positions.
ARROW_SCHEMA = pa.schema(
    [
        ("captured_cmd", pa.string()),
        ("captured_args", pa.string()),
        ("timestamp", pa.string()),
        ("source_ip", pa.string()),
        ("source_port", pa.int64()),
        ("user_agent", pa.string()),
        ("language", pa.string()),
        ("x_forwarded_for/real-ip", pa.string()),
    ]
)

COLS = [f.name for f in ARROW_SCHEMA]


def _normalize_nullable_text(v: Any) -> str | None:
    if v is None or pd.isna(v):
        return None
    if not isinstance(v, str):
        return None
    s = v.strip()
    if not s:
        return None
    if s.lower() in {"null", "none", "nan", "na"}:
        return None
    return s


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    for c in COLS:
        if c not in df.columns:
            df[c] = None

    df = df[COLS]

    text_cols = [
        "captured_cmd",
        "captured_args",
        "timestamp",
        "source_ip",
        "user_agent",
        "language",
        "x_forwarded_for/real-ip",
    ]
    for c in text_cols:
        df[c] = df[c].map(_normalize_nullable_text)

    df["source_port"] = pd.to_numeric(df["source_port"], errors="coerce").astype(
        "Int64"
    )
    return df


def write_events_to_parquet(
    events_iter,
    out_path: str,
    batch_size: int = 200_000,
) -> None:
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(out_path):
        os.remove(out_path)

    writer = None
    batch: List[Dict[str, Any]] = []
    total = 0

    for ev in tqdm(events_iter, desc="Ingesting", unit=" lines"):
        batch.append(ev)
        if len(batch) >= batch_size:
            df = _normalize_df(pd.DataFrame(batch))
            table = pa.Table.from_pandas(df, schema=ARROW_SCHEMA, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, ARROW_SCHEMA, compression="zstd")
            writer.write_table(table)
            total += len(batch)
            batch.clear()

    if batch:
        df = _normalize_df(pd.DataFrame(batch))
        table = pa.Table.from_pandas(df, schema=ARROW_SCHEMA, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, ARROW_SCHEMA, compression="zstd")
        writer.write_table(table)
        total += len(batch)

    if writer is not None:
        writer.close()

    print(f"Wrote {total} events -> {out_path}")
