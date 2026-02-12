from __future__ import annotations

from dataclasses import dataclass
import os
import uuid

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class SessionConfig:
    gap_seconds: int = 10 * 60  # 10 minutes
    min_events: int = 2  # ignore singletons if needed


def _ensure_event_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map = {
        "ts": "timestamp",
        "src_ip": "source_ip",
        "src_port": "source_port",
        "lang": "language",
        "action1": "captured_cmd",
        "action2": "captured_args",
        "extra": "x_forwarded_for/real-ip",
    }
    for old, new in rename_map.items():
        if old in out.columns and new not in out.columns:
            out = out.rename(columns={old: new})

    required = [
        "captured_cmd",
        "captured_args",
        "timestamp",
        "source_ip",
        "source_port",
        "user_agent",
        "language",
        "x_forwarded_for/real-ip",
        "tool",
    ]
    for c in required:
        if c not in out.columns:
            out[c] = None
    return out


def _ensure_dt(df: pd.DataFrame) -> pd.DataFrame:
    out = _ensure_event_schema(df)
    out["timestamp_dt"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp_dt", "source_ip"])
    return out


def build_sessions(
    events: pd.DataFrame,
    cfg: SessionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      sessions_df: one row per session
      events_df: original events + session_id
    """
    df = _ensure_dt(events)
    df = df.sort_values(
        ["source_ip", "tool", "timestamp_dt"], kind="mergesort"
    ).reset_index(drop=True)

    # session boundary when time gap exceeds threshold within same (source_ip, tool)
    df["prev_ts"] = df.groupby(["source_ip", "tool"])["timestamp_dt"].shift(1)
    df["gap_s"] = (df["timestamp_dt"] - df["prev_ts"]).dt.total_seconds()
    df["new_session"] = (df["prev_ts"].isna()) | (df["gap_s"] > cfg.gap_seconds)

    # session index per (source_ip, tool)
    df["session_idx"] = (
        df.groupby(["source_ip", "tool"])["new_session"].cumsum().astype("int64")
    )

    # stable session id string (uuid5 = deterministic for same key)
    def make_sid(row):
        base = f"{row['source_ip']}|{row['tool']}|{int(row['session_idx'])}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, base))

    keys = df[["source_ip", "tool", "session_idx"]].drop_duplicates()
    keys["session_id"] = keys.apply(make_sid, axis=1)
    df = df.merge(keys, on=["source_ip", "tool", "session_idx"], how="left")

    indicator_cols = [c for c in df.columns if c.startswith("ind_")]
    agg_spec: dict[str, tuple[str, object]] = {
        "source_ip": ("source_ip", "first"),
        "tool": ("tool", "first"),
        "start_ts": ("timestamp_dt", "min"),
        "end_ts": ("timestamp_dt", "max"),
        "event_count": ("timestamp_dt", "size"),
        "unique_ports": ("source_port", pd.Series.nunique),
        "indicator_hits": (
            (
                "indicator_count",
                "sum",
            )
            if "indicator_count" in df.columns
            else ("session_id", "size")
        ),
    }
    for col in indicator_cols:
        agg_spec[col] = (col, "sum")

    agg = df.groupby("session_id").agg(**agg_spec).reset_index()
    agg["duration_s"] = (
        (agg["end_ts"] - agg["start_ts"]).dt.total_seconds().clip(lower=0.0)
    )
    agg["rps"] = agg["event_count"] / (agg["duration_s"].replace(0, 1.0))

    # compatibility alias used by downstream mapping/rag/eval modules
    agg["src_ip"] = agg["source_ip"]

    if cfg.min_events > 1:
        keep = agg["event_count"] >= cfg.min_events
        keep_ids = set(agg.loc[keep, "session_id"])
        agg = agg.loc[keep].reset_index(drop=True)
        df = df[df["session_id"].isin(keep_ids)].reset_index(drop=True)

    df = df.drop(columns=["prev_ts", "gap_s", "new_session", "session_idx"]).rename(
        columns={"timestamp_dt": "timestamp_dt_utc"}
    )
    return agg, df


def write_session_outputs(
    sessions_df: pd.DataFrame,
    events_df: pd.DataFrame,
    out_dir: str,
    shard_rows: int = 250_000,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    sessions_path = os.path.join(out_dir, "sessions.parquet")
    events_dir = os.path.join(out_dir, "session_events")
    os.makedirs(events_dir, exist_ok=True)

    sessions_df.to_parquet(sessions_path, index=False)
    print(f"wrote {len(sessions_df)} sessions -> {sessions_path}")

    indicator_cols = [c for c in events_df.columns if c.startswith("ind_")]
    keep_cols = [
        "session_id",
        "timestamp",
        "timestamp_dt_utc",
        "source_ip",
        "source_port",
        "tool",
        "user_agent",
        "language",
        "captured_cmd",
        "captured_args",
        "x_forwarded_for/real-ip",
        "indicator_count",
    ] + indicator_cols

    for c in keep_cols:
        if c not in events_df.columns:
            events_df[c] = None
    events_df = events_df[keep_cols]

    schema = pa.Schema.from_pandas(events_df.head(1), preserve_index=False)
    part = 0
    buff = []

    for i in range(len(events_df)):
        buff.append(events_df.iloc[i])
        if len(buff) >= shard_rows:
            chunk = pd.DataFrame(buff)
            table = pa.Table.from_pandas(chunk, schema=schema, preserve_index=False)
            path = os.path.join(events_dir, f"part-{part:05d}.parquet")
            pq.write_table(table, path, compression="zstd")
            part += 1
            buff.clear()

    if buff:
        chunk = pd.DataFrame(buff)
        table = pa.Table.from_pandas(chunk, schema=schema, preserve_index=False)
        path = os.path.join(events_dir, f"part-{part:05d}.parquet")
        pq.write_table(table, path, compression="zstd")

    print(f"wrote session events shards -> {events_dir}")
