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
    min_events: int = 2  # ignore singletons if you want


def _ensure_dt(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # your ts is ISO-like "YYYY-mm-ddTHH:MM:SSZ"
    out["ts_dt"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    out = out.dropna(subset=["ts_dt", "src_ip"])
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

    # sort so sessionization is deterministic
    df = df.sort_values(["src_ip", "tool", "ts_dt"], kind="mergesort").reset_index(
        drop=True
    )

    # session boundary when time gap exceeds threshold within same (src_ip, tool)
    df["prev_ts"] = df.groupby(["src_ip", "tool"])["ts_dt"].shift(1)
    df["gap_s"] = (df["ts_dt"] - df["prev_ts"]).dt.total_seconds()
    df["new_session"] = (df["prev_ts"].isna()) | (df["gap_s"] > cfg.gap_seconds)

    # session index per (src_ip, tool)
    df["session_idx"] = (
        df.groupby(["src_ip", "tool"])["new_session"].cumsum().astype("int64")
    )

    # stable session id string
    # (uuid5 makes it stable/reproducible for same data)
    def make_sid(row):
        base = f"{row['src_ip']}|{row['tool']}|{int(row['session_idx'])}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, base))

    # compute once per group
    keys = df[["src_ip", "tool", "session_idx"]].drop_duplicates()
    keys["session_id"] = keys.apply(make_sid, axis=1)

    df = df.merge(keys, on=["src_ip", "tool", "session_idx"], how="left")

    # session aggregates (rate features here)
    agg = (
        df.groupby("session_id")
        .agg(
            src_ip=("src_ip", "first"),
            tool=("tool", "first"),
            start_ts=("ts_dt", "min"),
            end_ts=("ts_dt", "max"),
            event_count=("ts_dt", "size"),
            unique_ports=("src_port", pd.Series.nunique),
            indicator_hits=("indicator_count", "sum"),
            ind_lfi_etc_passwd=("ind_lfi_etc_passwd", "sum"),
            ind_path_traversal=("ind_path_traversal", "sum"),
            ind_sql_injection=("ind_sql_injection", "sum"),
            ind_cmd_injection=("ind_cmd_injection", "sum"),
            ind_wp_probe=("ind_wp_probe", "sum"),
        )
        .reset_index()
    )

    agg["duration_s"] = (
        (agg["end_ts"] - agg["start_ts"]).dt.total_seconds().clip(lower=0.0)
    )
    agg["rps"] = agg["event_count"] / (agg["duration_s"].replace(0, 1.0))

    # optional: filter very small sessions
    if cfg.min_events > 1:
        keep = agg["event_count"] >= cfg.min_events
        keep_ids = set(agg.loc[keep, "session_id"])
        agg = agg.loc[keep].reset_index(drop=True)
        df = df[df["session_id"].isin(keep_ids)].reset_index(drop=True)

    # final cleanup columns
    df = df.drop(columns=["prev_ts", "gap_s", "new_session", "session_idx"]).rename(
        columns={"ts_dt": "ts_dt_utc"}
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

    # sessions.parquet
    sessions_df.to_parquet(sessions_path, index=False)
    print(f"✅ wrote {len(sessions_df)} sessions -> {sessions_path}")

    # shard events into session_events/part-*.parquet
    # keep event columns compact
    keep_cols = [
        "session_id",
        "ts",
        "ts_dt_utc",
        "src_ip",
        "src_port",
        "tool",
        "user_agent",
        "action1",
        "action2",
        "extra",
        "indicator_count",
        "ind_lfi_etc_passwd",
        "ind_path_traversal",
        "ind_sql_injection",
        "ind_cmd_injection",
        "ind_wp_probe",
        "raw_line",
    ]
    for c in keep_cols:
        if c not in events_df.columns:
            events_df[c] = None
    events_df = events_df[keep_cols]

    # write shards with fixed schema for stability
    schema = pa.Schema.from_pandas(events_df.head(1), preserve_index=False)
    writer = None
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

    print(f"✅ wrote session events shards -> {events_dir}")
