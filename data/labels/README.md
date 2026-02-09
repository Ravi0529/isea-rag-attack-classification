# Phase 8 Labels

Use templates first:

```powershell
python -m src.cli.main eval-templates --sample-rows 1000
```

This creates:

- `data/labels/detection_labels.csv`
- `data/labels/mapping_labels.csv`

Fill required truth columns:

- `detection_labels.csv`: `true_label` in `{benign, suspicious, attack_like}`
- `mapping_labels.csv`: `true_technique_ids` (single `Txxxx` or multiple split by `|` or `,`)

Run labeled evaluation:

```powershell
python -m src.cli.main eval --mode labeled --detection-labels-path data/labels/detection_labels.csv --mapping-labels-path data/labels/mapping_labels.csv --retrieval-k 10
```

Run proxy evaluation (no manual labels):

```powershell
python -m src.cli.main eval --mode proxy
```

