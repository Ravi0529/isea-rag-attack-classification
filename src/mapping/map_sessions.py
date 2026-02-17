import pandas as pd
from typing import Optional
from tqdm import tqdm

from src.rag.retrieve import search_attack_for_text
from src.mapping.summary import session_to_summary
from src.mapping.scoring import compute_boost, final_score


def map_sessions(
    sessions_path: str,
    out_path: str,
    top_k: int = 20,
    embed_model: str = "sentence-transformers/all-roberta-large-v1",
    device: str = "auto",
    keep_top_n: int = 3,
    limit: Optional[int] = None,
):
    s = pd.read_parquet(sessions_path)

    s = s[s["label"].isin(["attack_like", "suspicious"])].copy()
    if limit is not None:
        s = s.head(limit)

    print(f"📊 Processing {len(s)} sessions...")
    mapped_rows = []
    for _, row in tqdm(s.iterrows(), total=len(s), desc="Mapping sessions", unit="session"):
        rowd = row.to_dict()
        summary = session_to_summary(rowd)

        hits = search_attack_for_text(
            summary,
            top_k=top_k,
            embed_model=embed_model,
            device=device,
        )

        boost = compute_boost(rowd)

        ranked = []
        for h in hits:
            p = h.payload
            sim = float(h.score)
            conf = final_score(sim, boost)
            ranked.append(
                {
                    "technique_id": p.get("technique_id"),
                    "name": p.get("name"),
                    "tactics": p.get("tactics"),
                    "sim": sim,
                    "confidence": conf,
                }
            )

        ranked = sorted(ranked, key=lambda x: x["confidence"], reverse=True)[
            :keep_top_n
        ]

        mapped_rows.append(
            {
                "session_id": rowd["session_id"],
                "src_ip": rowd.get("src_ip"),
                "label": rowd.get("label"),
                "tool": rowd.get("tool"),
                "suspicious_score": rowd.get("suspicious_score"),
                "start_ts": rowd.get("start_ts"),
                "summary": summary,
                "mapped_techniques": [x["technique_id"] for x in ranked],
                "mapped_names": [x["name"] for x in ranked],
                "mapped_tactics": [x["tactics"] for x in ranked],
                "confidence": ranked[0]["confidence"] if ranked else 0.0,
                "why": (
                    rowd.get("rule_reasons").tolist()
                    if hasattr(rowd.get("rule_reasons"), "tolist")
                    else rowd.get("rule_reasons", [])
                ),
            }
        )

    out = pd.DataFrame(mapped_rows)
    out.to_parquet(out_path, index=False)
    print(f"✅ Completed! Mapped {len(out)} sessions -> {out_path}")
    return out
