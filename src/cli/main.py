from __future__ import annotations
import os
import typer

from src.ingest.parser import iter_events
from src.ingest.write_parquet import write_events_to_parquet

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


app()
