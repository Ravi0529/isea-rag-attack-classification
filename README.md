# ISEA RAG Attack Classification

End-to-end cybersecurity pipeline for:

- raw log ingestion and sessionization,
- hybrid anomaly detection (rules + embedding-based ML),
- MITRE ATT&CK retrieval and tactic/technique mapping (RAG style),
- evaluation and analyst-facing outputs.

## What This Project Does

The pipeline converts raw logs into structured session intelligence and maps suspicious sessions to MITRE ATT&CK techniques.

Core outputs:

- `data/processed/sessions_scored.parquet` (detection output)
- `data/processed/session_attack_mapping.parquet` (ATT&CK mapping output)
- `reports/metrics.json` + `reports/figures/*` (evaluation)
- `notebooks/phase9_final_source_ip_tactic_technique_outputs.ipynb` (final source IP -> tactic -> technique summaries)

## Pipeline Phases

1. **Phase 1**: Ingest raw logs -> `events.parquet`
2. **Phase 2**: Enrich + sessionize -> `sessions.parquet`, `session_events/*`
3. **Phase 3**: Detect suspicious sessions -> `sessions_scored.parquet`
4. **Phase 4**: Download/build ATT&CK cache -> `attack_stix_cache.json`
5. **Phase 5**: Index ATT&CK + sessions into Qdrant
6. **Phase 6**: Map suspicious sessions to ATT&CK -> `session_attack_mapping.parquet`
7. **Phase 7**: Generate manual-label templates (optional)
8. **Phase 8**: Evaluate (proxy or labeled) -> metrics + figures
9. **Phase 9**: Final reporting notebook with source IP to tactic-technique outputs and plots

## Repository Layout

- `src/cli/main.py` - all CLI commands
- `src/ingest/*` - parsing + parquet writing
- `src/features/*` - enrichment + sessionization
- `src/detection/*` - rules, embeddings, anomaly scoring, score fusion
- `src/mitre/*` - ATT&CK STIX download/cache build
- `src/rag/*` - embeddings, Qdrant indexing, retrieval
- `src/mapping/*` - session-to-ATT&CK mapping
- `src/eval/*` - metrics, plots, evaluation runner
- `notebooks/phase*.ipynb` - phase-wise analysis/visuals

## Prerequisites

- Python 3.10+
- Docker (recommended for Qdrant)
- Windows PowerShell examples are used below

## Setup

```powershell
# from repo root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e .
```

Optional `.env`:

```env
QDRANT_URL=http://localhost:6333
RAW_LOG_PATH=./data/raw/cj.log
OUT_DIR=./data/processed
```

## Start Qdrant

```powershell
docker compose -f docker/docker-compose.yaml up -d qdrant
```

## CLI Quick Help

```powershell
python -m src.cli.main --help
```

---

## Run Every Phase (CLI)

### Phase 1 - Ingest raw log file

```powershell
python -m src.cli.main ingest `
  --raw-path data/raw/cj.log `
  --out-dir data/processed
```

Output: `data/processed/events.parquet`

### Phase 2 - Enrich + sessionize

```powershell
python -m src.cli.main sessionize `
  --events-path data/processed/events.parquet `
  --out-dir data/processed `
  --gap-seconds 600 `
  --min-events 2
```

Outputs:

- `data/processed/sessions.parquet`
- `data/processed/session_events/part-*.parquet`

### Phase 3 - Hybrid detection

```powershell
python -m src.cli.main detect `
  --sessions-path data/processed/sessions.parquet `
  --out-path data/processed/sessions_scored.parquet `
  --embed-model BAAI/bge-large-en-v1.5 `
  --device auto
```

Output: `data/processed/sessions_scored.parquet`

Use LOF instead of Isolation Forest:

```powershell
python -m src.cli.main detect --use-lof
```

### Phase 4a - Download ATT&CK STIX

```powershell
python -m src.cli.main attack-download `
  --out-path data/attack/raw/enterprise-attack.json
```

### Phase 4b - Build ATT&CK cache/index JSON

```powershell
python -m src.cli.main attack-cache `
  --stix-path data/attack/raw/enterprise-attack.json `
  --out-dir data/attack
```

Outputs:

- `data/attack/attack_stix_cache.json`
- `data/attack/indexes/techniques_by_tid.json`
- `data/attack/indexes/objects_by_stix_id.json`

### Phase 5a - Index ATT&CK techniques in Qdrant

```powershell
python -m src.cli.main qdrant-index-attack `
  --cache-path data/attack/attack_stix_cache.json `
  --embed-model BAAI/bge-large-en-v1.5 `
  --device auto
```

### Phase 5b - Index scored sessions in Qdrant

```powershell
python -m src.cli.main qdrant-index-sessions `
  --scored-sessions-path data/processed/sessions_scored.parquet `
  --embed-model BAAI/bge-large-en-v1.5 `
  --device auto
```

### Phase 5c - Test ATT&CK retrieval (optional)

```powershell
python -m src.cli.main rag-attack-search `
  --q "high-rate command injection behavior with automated tool" `
  --top-k 8
```

### Phase 6 - Map suspicious sessions to ATT&CK

```powershell
python -m src.cli.main map-techniques `
  --sessions-path data/processed/sessions_scored.parquet `
  --out-path data/processed/session_attack_mapping.parquet `
  --top-k 20 `
  --keep-top-n 3 `
  --embed-model BAAI/bge-large-en-v1.5 `
  --device auto
```

Output: `data/processed/session_attack_mapping.parquet`

### Phase 7 - Create manual label templates (optional)

```powershell
python -m src.cli.main eval-templates `
  --sessions-scored-path data/processed/sessions_scored.parquet `
  --session-mapping-path data/processed/session_attack_mapping.parquet `
  --out-dir data/labels `
  --sample-rows 1000 `
  --stratified true
```

Outputs:

- `data/labels/detection_labels.csv`
- `data/labels/mapping_labels.csv`

### Phase 8 - Evaluate (proxy mode)

```powershell
python -m src.cli.main eval `
  --mode proxy `
  --sessions-scored-path data/processed/sessions_scored.parquet `
  --session-mapping-path data/processed/session_attack_mapping.parquet `
  --attack-cache-path data/attack/attack_stix_cache.json `
  --out-json-path reports/metrics.json `
  --figures-dir reports/figures
```

### Phase 8 - Evaluate (labeled mode)

```powershell
python -m src.cli.main eval `
  --mode labeled `
  --sessions-scored-path data/processed/sessions_scored.parquet `
  --session-mapping-path data/processed/session_attack_mapping.parquet `
  --attack-cache-path data/attack/attack_stix_cache.json `
  --detection-labels-path data/labels/detection_labels.csv `
  --mapping-labels-path data/labels/mapping_labels.csv
```

### Phase 9 - Final notebook outputs

Open and run:

- `notebooks/phase9_final_source_ip_tactic_technique_outputs.ipynb`

This notebook produces final analyst-facing summaries:

- `source_ip`, `tactic`, `technique_name`, `technique_id`, `hits`, `first_seen`, `last_seen`
- plots for top tactics, top techniques, strongest source-IP connections, and embedding model comparison.

---

## Reproducing Embedding Model Comparison (Detection F1)

Run Phase 3 + Phase 8 separately for each embedding model and record weighted F1 from `reports/metrics.json`.

Examples:

```powershell
# RoBERTa
python -m src.cli.main detect --embed-model sentence-transformers/all-roberta-large-v1
python -m src.cli.main eval --mode proxy

# BGE
python -m src.cli.main detect --embed-model BAAI/bge-large-en-v1.5
python -m src.cli.main eval --mode proxy

# MPNet
python -m src.cli.main detect --embed-model sentence-transformers/all-mpnet-base-v2
python -m src.cli.main eval --mode proxy
```

---

## Notebooks by Phase

- `notebooks/phase1_ingest_debug.ipynb`
- `notebooks/phase2_enrich_sessionize.ipynb`
- `notebooks/phase3_detection_score.ipynb`
- `notebooks/phase4_stix_dataset_json.ipynb`
- `notebooks/phase5_qdrant_rag_sessions_and_attack.ipynb`
- `notebooks/phase6_session_attack_mapping.ipynb`
- `notebooks/phase7_evaluation.ipynb`
- `notebooks/phase8_detailed_evaluation.ipynb`
- `notebooks/phase9_final_source_ip_tactic_technique_outputs.ipynb`

## Notes

- Keep embedding model dimensions consistent between ATT&CK indexing and mapping queries.
- Start Qdrant before running indexing/retrieval/mapping commands.
- If GPU is unavailable, use `--device cpu`.
- Do not commit private API/HF tokens in `.env`.
