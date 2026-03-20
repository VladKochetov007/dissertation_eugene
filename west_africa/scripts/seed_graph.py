"""Validate and summarize the West Africa graph seed data.

Usage:
    python -m west_africa.scripts.seed_graph
"""

from __future__ import annotations

import json
import pathlib
import sys

# Ensure project root is on sys.path
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from west_africa.core.graph import WestAfricaGraph, DATA_DIR
from west_africa.core.types import BlocMembership, ConnectionType


def _header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def validate_edges(data_dir: pathlib.Path) -> list[str]:
    """Check that every edge references an existing city id."""
    with open(data_dir / "cities.json") as f:
        cities = {c["id"] for c in json.load(f)}
    with open(data_dir / "edges.json") as f:
        edges = json.load(f)

    errors: list[str] = []
    for i, e in enumerate(edges):
        src, tgt = e.get("source", ""), e.get("target", "")
        if src not in cities:
            errors.append(f"Edge {i}: source '{src}' not in cities")
        if tgt not in cities:
            errors.append(f"Edge {i}: target '{tgt}' not in cities")
        # Validate edge_type enum
        etype = e.get("edge_type", "")
        try:
            ConnectionType(etype)
        except ValueError:
            errors.append(f"Edge {i}: unknown edge_type '{etype}'")
    return errors


def validate_economic_state(data_dir: pathlib.Path) -> list[str]:
    """Check that every economic_state entry references an existing city id."""
    with open(data_dir / "cities.json") as f:
        cities = {c["id"] for c in json.load(f)}
    with open(data_dir / "economic_state.json") as f:
        states = json.load(f)

    errors: list[str] = []
    for i, es in enumerate(states):
        cid = es.get("city_id", "")
        if cid not in cities:
            errors.append(f"EconomicState {i}: city_id '{cid}' not in cities")
        # Validate bloc_override if present
        override = es.get("bloc_override")
        if override:
            try:
                BlocMembership(override)
            except ValueError:
                errors.append(f"EconomicState {i}: unknown bloc_override '{override}'")
    return errors


def print_summary(graph: WestAfricaGraph) -> None:
    """Print a full summary of the loaded graph."""
    summary = graph.summary()

    _header("GRAPH SUMMARY")
    print(f"  Total cities (nodes):  {summary['nodes']}")
    print(f"  Total edges:           {summary['edges']}")
    print(f"  Economic states:       {len(graph.economic_states)}")

    _header("BLOC DISTRIBUTION")
    blocs = [
        ("ECOWAS (active, non-CFA)", BlocMembership.ECOWAS),
        ("UEMOA (CFA zone)",         BlocMembership.UEMOA),
        ("SUSPENDED",                BlocMembership.SUSPENDED),
        ("EXTERNAL",                 BlocMembership.EXTERNAL),
    ]
    for label, bloc in blocs:
        cities = graph.get_by_bloc(bloc)
        city_names = ", ".join(c.name for c in cities[:8])
        suffix = f" (+{len(cities) - 8} more)" if len(cities) > 8 else ""
        print(f"  {label:30s}  {len(cities):3d}  {city_names}{suffix}")

    _header("EDGE TYPE DISTRIBUTION")
    edge_type_counts: dict[str, int] = {}
    edge_type_volume: dict[str, float] = {}
    for _, _, _, data in graph.G.edges(keys=True, data=True):
        etype = data.get("edge_type", "UNKNOWN")
        edge_type_counts[etype] = edge_type_counts.get(etype, 0) + 1
        edge_type_volume[etype] = edge_type_volume.get(etype, 0.0) + data.get("volume", 0.0)
    print(f"  {'Type':20s} {'Count':>6s} {'Total Volume (USD M)':>22s}")
    print(f"  {'-' * 20} {'-' * 6} {'-' * 22}")
    for etype in sorted(edge_type_counts.keys()):
        count = edge_type_counts[etype]
        vol = edge_type_volume[etype]
        print(f"  {etype:20s} {count:6d} {vol:22.1f}")

    _header("PORT CITIES")
    ports = graph.get_port_cities()
    for p in sorted(ports, key=lambda c: c.name):
        print(f"  {p.name:25s}  {p.country:20s}  ({p.country_iso3})")
    print(f"  Total: {len(ports)}")

    _header("FTZ TARGET CITIES")
    targets = graph.get_ftz_targets()
    for t in sorted(targets, key=lambda c: c.name):
        bloc = graph.get_effective_bloc(t.id)
        port_flag = " [PORT]" if t.is_port else ""
        cap_flag = " [CAPITAL]" if t.is_capital else ""
        print(f"  {t.name:25s}  {t.country:20s}  {bloc.value:10s}{port_flag}{cap_flag}")
    print(f"  Total: {len(targets)}")


def main() -> None:
    print("West Africa FTZ -- Seed Data Validator")
    print(f"Data directory: {DATA_DIR}")

    # Validation
    _header("VALIDATION")
    edge_errors = validate_edges(DATA_DIR)
    econ_errors = validate_economic_state(DATA_DIR)

    all_errors = edge_errors + econ_errors
    if all_errors:
        print(f"  ERRORS FOUND: {len(all_errors)}")
        for err in all_errors:
            print(f"    - {err}")
    else:
        print("  All edges reference existing cities ......... OK")
        print("  All economic states reference existing cities  OK")
        print("  All enum values are valid ................... OK")

    # Load and print
    graph = WestAfricaGraph.from_seed_data()
    print_summary(graph)

    if all_errors:
        print(f"\n** {len(all_errors)} validation error(s) found. Fix seed data before proceeding. **")
        sys.exit(1)
    else:
        print("\nSeed data is valid and consistent.")


if __name__ == "__main__":
    main()
