from __future__ import annotations
import os
import typer
import pandas as pd
from typing import Optional

from src.ingest.parser import iter_events
from src.ingest.write_parquet import write_events_to_parquet

from src.features.enrich import enrich_events
from src.features.sessionize import SessionConfig, build_sessions, write_session_outputs

from src.detection.rules import apply_rules
from src.detection.embedding_anomaly import (
    embed_sessions,
    isolation_forest_scores,
    lof_scores,
)
from src.detection.score import combine_scores

from src.mitre.download_attack import download_enterprise_attack
from src.mitre.stix_cache import build_attack_cache

from src.rag.index_attack import index_mitre_attack
from src.rag.index_sessions import index_log_sessions
from src.rag.retrieve import search_attack_for_text

from src.mapping.map_sessions import map_sessions

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


@app.command()
def detect(
    sessions_path: str = typer.Option(default="data/processed/sessions.parquet"),
    out_path: str = typer.Option(default="data/processed/sessions_scored.parquet"),
    embed_model: str = typer.Option(default="sentence-transformers/all-mpnet-base-v2"),
    use_lof: bool = typer.Option(default=False),
    device: str = typer.Option(default="auto"),
):
    s = pd.read_parquet(sessions_path)
    s = apply_rules(s)

    X = embed_sessions(s, model_name=embed_model, device=device)
    s["ml_score"] = lof_scores(X) if use_lof else isolation_forest_scores(X)

    s = combine_scores(s)
    s.to_parquet(out_path, index=False)
    print(f"✅ wrote scored sessions -> {out_path}")
    print(s["label"].value_counts().to_string())


@app.command()
def attack_download(
    out_path: str = typer.Option(default="data/attack/raw/enterprise-attack.json"),
):
    download_enterprise_attack(out_path)


@app.command()
def attack_cache(
    stix_path: str = typer.Option(default="data/attack/raw/enterprise-attack.json"),
    out_dir: str = typer.Option(default="data/attack"),
):
    build_attack_cache(stix_path, out_dir)


@app.command()
def qdrant_index_attack(
    cache_path: str = typer.Option(default="data/attack/attack_stix_cache.json"),
    embed_model: str = typer.Option(default="sentence-transformers/all-mpnet-base-v2"),
    device: str = typer.Option(default="auto"),
):
    index_mitre_attack(cache_path=cache_path, embed_model=embed_model, device=device)


@app.command()
def qdrant_index_sessions(
    scored_sessions_path: str = typer.Option(
        default="data/processed/sessions_scored.parquet"
    ),
    embed_model: str = typer.Option(default="sentence-transformers/all-mpnet-base-v2"),
    device: str = typer.Option(default="auto"),
):
    index_log_sessions(
        scored_sessions_path=scored_sessions_path,
        embed_model=embed_model,
        device=device,
    )


@app.command()
def rag_attack_search(
    q: str = typer.Option(...),
    top_k: int = typer.Option(default=8),
    embed_model: str = typer.Option(default="sentence-transformers/all-mpnet-base-v2"),
    device: str = typer.Option(default="auto"),
):
    hits = search_attack_for_text(
        q, top_k=top_k, embed_model=embed_model, device=device
    )
    for h in hits:
        p = h.payload
        print(
            f"{h.score:.3f}  {p.get('technique_id')}  {p.get('name')}  tactics={p.get('tactics')}"
        )


@app.command("map-techniques")
def map_techniques(
    sessions_path: str = typer.Option(..., help="Path to sessions_scored.parquet"),
    out_path: str = typer.Option(..., help="Output parquet path for mapping results"),
    top_k: int = typer.Option(
        20, help="How many ATT&CK candidates to retrieve per session"
    ),
    keep_top_n: int = typer.Option(3, help="How many techniques to keep per session"),
    embed_model: str = typer.Option(
        "sentence-transformers/all-mpnet-base-v2",
        help="Embedding model (must match Qdrant dims)",
    ),
    device: str = typer.Option("auto", help="cpu/cuda/auto"),
    limit: Optional[int] = typer.Option(
        None, help="Process only first N sessions (debug/demo)"
    ),
):
    """
    Map suspicious sessions -> MITRE ATT&CK techniques (explainable).
    Writes: session_attack_mapping.parquet
    """
    map_sessions(
        sessions_path=sessions_path,
        out_path=out_path,
        top_k=top_k,
        embed_model=embed_model,
        device=device,
        keep_top_n=keep_top_n,
        limit=limit,
    )
    print(f"✅ wrote mapping -> {out_path}")


app()
