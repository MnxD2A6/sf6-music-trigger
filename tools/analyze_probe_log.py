"""Summarise a REFramework probe log (probe-*.jsonl) from a training-mode session.

Usage:
    python tools/analyze_probe_log.py <probe-YYYYMMDD-HHMMSS.jsonl> [--json out.json]

The report answers the Phase-A questions:
  * Were character ids and super tables read for both sides?
  * Did every super start produce exactly one super_start (no duplicates)?
  * What did the opponent do (guard / damage / none) and how did it end?
  * Which cPlayer field dropped by exactly the super's gauge cost (super gauge)?
  * Which opponent fields dropped on hits (vital candidates)?
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_events(path: Path) -> list[dict]:
    events = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            events.append({"type": "corrupt_line", "line": line_no})
    return events


def build_episodes(events: list[dict]) -> list[dict]:
    """Pair super_start with its diff, reactions and end, per side."""
    open_by_side: dict[int, dict] = {}
    episodes = []
    for e in events:
        kind = e.get("type")
        side = e.get("side")
        if kind == "super_start":
            ep = {"side": side, "label": e["label"], "act": e["act"], "chara": e.get("chara"),
                  "gauge_consume": e.get("gauge_consume", 0), "t": e["t"],
                  "reactions": [], "diff": None, "end": None, "continues": 0}
            open_by_side[side] = ep
            episodes.append(ep)
        elif side in open_by_side:
            ep = open_by_side[side]
            if kind == "opp_reaction":
                ep["reactions"].append((round(e["since_start"], 3), e["class"], e.get("opp_name")))
            elif kind == "super_diff":
                ep["diff"] = e
            elif kind == "super_continue":
                ep["continues"] += 1
            elif kind == "super_end":
                ep["end"] = e
                del open_by_side[side]
    return episodes


def gauge_candidates(episodes: list[dict]) -> Counter:
    """Fields on the attacker that dropped by exactly the trigger's gauge cost."""
    hits = Counter()
    for ep in episodes:
        cost = ep["gauge_consume"]
        diff = (ep["diff"] or {}).get("self_diff") or {}
        if not cost:
            continue
        for name, pair in diff.items():
            old, new = pair if len(pair) == 2 else (None, None)
            if isinstance(old, int) and isinstance(new, int) and old - new == cost:
                hits[name] += 1
    return hits


def vital_candidates(episodes: list[dict]) -> Counter:
    """Opponent fields that decreased during supers the opponent reacted to with damage."""
    hits = Counter()
    for ep in episodes:
        classes = {r[1] for r in ep["reactions"]}
        if "damage" not in classes:
            continue
        diff = (ep["end"] or {}).get("opp_diff") or (ep["diff"] or {}).get("opp_diff") or {}
        for name, pair in diff.items():
            old, new = pair if len(pair) == 2 else (None, None)
            if isinstance(old, int) and isinstance(new, int) and new < old:
                hits[name] += 1
    return hits


def summarise(events: list[dict]) -> dict:
    by_type = Counter(e.get("type") for e in events)
    session = next((e for e in events if e.get("type") == "session"), {})
    charas = defaultdict(list)
    for e in events:
        if e.get("type") == "chara":
            charas[e["side"]].append(e["chara"])
    tables = defaultdict(list)
    for e in events:
        if e.get("type") == "super_table":
            tables[e["side"]].append({"chara": e["chara"], "count": e["count"],
                                      "levels": sorted({(r["act"], r["level"]) for r in e["rows"]})})
    episodes = build_episodes(events)
    interesting = defaultdict(lambda: defaultdict(set))
    for e in events:
        if e.get("type") == "heartbeat":
            for side_key in ("p1_interesting", "p2_interesting"):
                for name, value in (e.get(side_key) or {}).items():
                    interesting[side_key][name].add(value)
    errors = [e for e in events if e.get("type") == "error"]
    return {
        "probe": session.get("probe"),
        "session": session.get("session"),
        "counts": dict(by_type),
        "charas": dict(charas),
        "super_tables": dict(tables),
        "episodes": [
            {
                "side": ep["side"], "label": ep["label"], "act": ep["act"], "chara": ep["chara"],
                "reactions": ep["reactions"], "continues": ep["continues"],
                "end_reason": (ep["end"] or {}).get("reason"),
                "opp_class": (ep["end"] or {}).get("opp_class"),
                "duration": round((ep["end"] or {}).get("duration", 0.0), 3),
            }
            for ep in episodes
        ],
        "gauge_candidates": gauge_candidates(episodes).most_common(10),
        "vital_candidates": vital_candidates(episodes).most_common(10),
        "side_fields": {k: {n: sorted(v)[:8] for n, v in fields.items()} for k, fields in interesting.items()},
        "errors": errors[:20],
    }


def render(summary: dict) -> str:
    lines = [f"probe={summary['probe']} session={summary['session']}",
             f"event counts: {summary['counts']}", ""]
    for side, charas in sorted(summary["charas"].items()):
        lines.append(f"P{side} characters seen: {charas}")
    for side, tables in sorted(summary["super_tables"].items()):
        for t in tables:
            lines.append(f"P{side} chara {t['chara']}: {t['count']} super actions {t['levels']}")
    lines.append("")
    lines.append("Super episodes:")
    for ep in summary["episodes"]:
        lines.append(f"  P{ep['side']} {ep['label']:<3} act={ep['act']} opp={ep['opp_class']} "
                     f"end={ep['end_reason']} dur={ep['duration']}s continues={ep['continues']} "
                     f"reactions={ep['reactions']}")
    lines.append("")
    lines.append(f"Super-gauge field candidates: {summary['gauge_candidates']}")
    lines.append(f"Vital field candidates: {summary['vital_candidates']}")
    lines.append(f"Side/CPU-related fields (heartbeat values): {summary['side_fields']}")
    if summary["errors"]:
        lines.append("")
        lines.append("Errors:")
        for e in summary["errors"]:
            lines.append(f"  {e.get('where')}: {e.get('msg')}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", type=Path)
    parser.add_argument("--json", type=Path, help="also write the summary as JSON")
    args = parser.parse_args()
    summary = summarise(load_events(args.log))
    print(render(summary))
    if args.json:
        args.json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
