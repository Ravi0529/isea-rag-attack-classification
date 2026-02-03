from __future__ import annotations
import os
import typer
import pandas as pd

from src.ingest.parser import iter_events
from src.ingest.write_parquet import write_events_to_parquet

from src.features.enrich import enrich_events
from src.features.sessionize import SessionConfig, build_sessions, write_session_outputs

app = typer.Typer(no_args_is_help=True)


@app.command()
def ingest(
    raw_path: str = typer.Option(default=os.getenv("RAW_LOG_PATH", "data/raw/cj.log")),
    out_dir: str = typer.Option(default=os.getenv("OUT_DIR", "data/processed")),
    batch_size: int = typer.Option(default=200_000),
):
    out_path = os.path.join(out_dir, "events.parquet")
    events = iter_events(raw_path)
    write_events_to_parquet(events, out_path=out_path, batch_size=batch_size)


@app.command()
def sessionize(
    events_path: str = typer.Option(default="data/processed/events.parquet"),
    out_dir: str = typer.Option(default="data/processed"),
    gap_seconds: int = typer.Option(default=600),
    min_events: int = typer.Option(default=2),
):
    df = pd.read_parquet(events_path)
    df = enrich_events(df)
    sessions_df, events_df = build_sessions(
        df, SessionConfig(gap_seconds=gap_seconds, min_events=min_events)
    )
    write_session_outputs(sessions_df, events_df, out_dir=out_dir)


app()
