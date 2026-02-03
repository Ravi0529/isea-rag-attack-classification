from __future__ import annotations
import os
from typing import Dict, Any, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Define a fixed schema (stable across all batches)
ARROW_SCHEMA = pa.schema(
    [
        ("ts", pa.string()),
        ("src_ip", pa.string()),
        ("src_port", pa.int64()),
        ("user_agent", pa.string()),
        ("lang", pa.string()),
        ("action1", pa.string()),
        ("action2", pa.string()),
        ("extra", pa.string()),
        ("raw_line", pa.string()),
    ]
)

COLS = [f.name for f in ARROW_SCHEMA]


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure all columns exist
    for c in COLS:
        if c not in df.columns:
            df[c] = None

    # Reorder
    df = df[COLS]

    # Make all string cols consistent (nullable string dtype)
    for c in [
        "ts",
        "src_ip",
        "user_agent",
        "lang",
        "action1",
        "action2",
        "extra",
        "raw_line",
    ]:
        df[c] = df[c].astype("string")

    # Keep src_port numeric nullable
    df["src_port"] = pd.to_numeric(df["src_port"], errors="coerce").astype("Int64")

    return df


def write_events_to_parquet(
    events_iter,
    out_path: str,
    batch_size: int = 200_000,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # If a partial file exists from the previous crash, remove it
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

    print(f"✅ Wrote {total} events -> {out_path}")
