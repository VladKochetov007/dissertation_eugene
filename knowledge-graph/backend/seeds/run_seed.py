"""
Seed runner — populates the knowledge graph with intellectual foundation data.

Usage:
    cd knowledge-graph/backend
    python -m seeds.run_seed

Generates a knowledge_base.json snapshot that can be loaded at API startup.
"""

import json
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.store import KnowledgeGraphStore
from seeds.domains import create_domains
from seeds.traditions import create_traditions
from seeds.historical_periods import create_historical_periods
from seeds.thinkers import create_thinkers
from seeds.concepts import create_concepts
from seeds.works import create_works


def seed_knowledge_graph() -> KnowledgeGraphStore:
    """Populate a knowledge graph store with the full intellectual foundation."""
    store = KnowledgeGraphStore()

    # Phase 1: Independent entities (no cross-references needed)
    print("Phase 1: Domains...")
    domains = create_domains()
    counts = store.bulk_add(domains)
    print(f"  Added {counts}")

    print("Phase 2: Traditions...")
    traditions = create_traditions()
    counts = store.bulk_add(traditions)
    print(f"  Added {counts}")

    print("Phase 3: Historical periods...")
    periods = create_historical_periods()
    counts = store.bulk_add(periods)
    print(f"  Added {counts}")

    # Phase 2: Entities referencing Phase 1
    print("Phase 4: Thinkers...")
    thinkers = create_thinkers()
    counts = store.bulk_add(thinkers)
    print(f"  Added {counts}")

    # Phase 3: Entities referencing Phase 1 + 2
    print("Phase 5: Concepts...")
    concepts = create_concepts()
    counts = store.bulk_add(concepts)
    print(f"  Added {counts}")

    print("Phase 6: Works...")
    works = create_works()
    counts = store.bulk_add(works)
    print(f"  Added {counts}")

    # Phase 4: Resolve deferred edges
    print("Phase 7: Resolving deferred edges...")
    new_edges = store.resolve_deferred_edges()
    print(f"  Created {new_edges} deferred edges")

    # Summary
    stats = store.stats()
    print("\n=== Knowledge Graph Summary ===")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    return store


def export_snapshot(store: KnowledgeGraphStore, path: Path) -> None:
    """Export the store to a JSON snapshot file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(store.to_json())
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"\nSnapshot saved to {path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    store = seed_knowledge_graph()

    snapshot_path = Path(__file__).parent / "knowledge_base.json"
    export_snapshot(store, snapshot_path)
