def session_to_summary(row: dict) -> str:
    reasons = row.get("rule_reasons", [])
    return (
        f"tool={row.get('tool')} label={row.get('label')} "
        f"src_ip={row.get('src_ip')} event_count={row.get('event_count')} "
        f"rps={row.get('rps')} unique_ports={row.get('unique_ports')} "
        f"indicator_hits={row.get('indicator_hits')} "
        f"reasons={reasons}"
    )
