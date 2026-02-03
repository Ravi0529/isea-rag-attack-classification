from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Optional


def _external_tid(obj: Dict[str, Any]) -> Optional[str]:
    """Return technique external_id like T1059 from external_references."""
    for ref in obj.get("external_references", []) or []:
        if ref.get("source_name") == "mitre-attack" and isinstance(
            ref.get("external_id"), str
        ):
            eid = ref["external_id"]
            if eid.startswith("T"):
                return eid
    return None


def _tactics_from_kill_chain(obj: Dict[str, Any]) -> List[str]:
    """
    ATT&CK tactics come via kill_chain_phases with kill_chain_name="mitre-attack"
    phase_name is the tactic (e.g., discovery, execution, initial-access).
    """
    tacs = []
    for k in obj.get("kill_chain_phases", []) or []:
        if k.get("kill_chain_name") == "mitre-attack" and k.get("phase_name"):
            tacs.append(k["phase_name"])
    return sorted(set(tacs))


def build_attack_cache(stix_path: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    idx_dir = os.path.join(out_dir, "indexes")
    os.makedirs(idx_dir, exist_ok=True)

    with open(stix_path, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    objects = bundle.get("objects", [])
    if not isinstance(objects, list):
        raise ValueError("Invalid STIX: missing objects[]")

    objects_by_id: Dict[str, Dict[str, Any]] = {}
    techniques_by_tid: Dict[str, Dict[str, Any]] = {}

    # store raw objects by stix id
    for obj in objects:
        if isinstance(obj, dict) and isinstance(obj.get("id"), str):
            objects_by_id[obj["id"]] = obj

    # extract techniques (attack-pattern) + tactics
    techniques: List[Dict[str, Any]] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        if obj.get("type") != "attack-pattern":
            continue

        tid = _external_tid(obj)
        tactics = _tactics_from_kill_chain(obj)

        tech = {
            "stix_id": obj.get("id"),
            "technique_id": tid,  # T####
            "name": obj.get("name"),
            "description": obj.get("description"),
            "tactics": tactics,
            "is_subtechnique": bool(obj.get("x_mitre_is_subtechnique", False)),
            "platforms": obj.get("x_mitre_platforms", []),
        }
        techniques.append(tech)

        if tid:
            techniques_by_tid[tid] = tech

    # mitigations (course-of-action)
    mitigations: Dict[str, Dict[str, Any]] = {}
    for obj in objects:
        if isinstance(obj, dict) and obj.get("type") == "course-of-action":
            mitigations[obj["id"]] = {
                "stix_id": obj.get("id"),
                "name": obj.get("name"),
                "description": obj.get("description"),
            }

    # relationships
    # mitigates: course-of-action -> attack-pattern
    mitigations_by_technique_tid: Dict[str, List[Dict[str, Any]]] = {}
    for obj in objects:
        if not isinstance(obj, dict) or obj.get("type") != "relationship":
            continue
        rel_type = obj.get("relationship_type")
        src = obj.get("source_ref")
        tgt = obj.get("target_ref")

        if rel_type == "mitigates" and src in mitigations and tgt in objects_by_id:
            target_obj = objects_by_id[tgt]
            tid = _external_tid(target_obj)
            if tid:
                mitigations_by_technique_tid.setdefault(tid, []).append(
                    mitigations[src]
                )

    # compact final cache
    cache = {
        "source": "mitre-attack/attack-stix-data enterprise-attack.json (STIX 2.1)",
        "techniques": techniques,
        "mitigations_by_technique": mitigations_by_technique_tid,
    }

    cache_path = os.path.join(out_dir, "attack_stix_cache.json")
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    # indexes (handy later)
    with open(
        os.path.join(idx_dir, "techniques_by_tid.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(techniques_by_tid, f, ensure_ascii=False, indent=2)

    with open(
        os.path.join(idx_dir, "objects_by_stix_id.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(objects_by_id, f, ensure_ascii=False, indent=2)

    print(f"✅ wrote cache -> {cache_path}")
    print(f"✅ wrote indexes -> {idx_dir}")
    return cache_path
