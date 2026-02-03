from __future__ import annotations
import os
import requests

RAW_URL = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json"


def download_enterprise_attack(out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    r = requests.get(RAW_URL, timeout=60)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)
    print(f"✅ downloaded enterprise-attack.json -> {out_path}")
