"""Generate all visualizations and the combined dashboard.

Usage:
    python -m west_africa.scripts.generate_viz [--output-dir OUTPUT_DIR]
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate West Africa FTZ visualizations")
    parser.add_argument(
        "--output-dir", "-o",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent.parent / "viz" / "output",
        help="Directory for generated HTML files",
    )
    args = parser.parse_args()

    print("West Africa FTZ -- Visualization Generator")
    print("=" * 60)

    t0 = time.time()

    # ── Load graph & metrics ────────────────────────────────
    print("\n  Loading graph...")
    from west_africa.core.graph import WestAfricaGraph
    from west_africa.core.metrics import GraphMetrics

    wag = WestAfricaGraph.from_seed_data()
    metrics = GraphMetrics(wag)
    summary = wag.summary()
    print(f"  Graph loaded: {summary['nodes']} nodes, {summary['edges']} edges")

    # ── Build dashboard ─────────────────────────────────────
    print(f"\n  Generating visualizations to: {args.output_dir}")
    from west_africa.viz.dashboard import DashboardBuilder

    builder = DashboardBuilder(wag, metrics)
    index_path = builder.build(args.output_dir)

    elapsed = time.time() - t0
    print(f"\n  Dashboard generated in {elapsed:.1f}s")
    print(f"  Open: {index_path}")

    # List generated files
    html_files = sorted(args.output_dir.glob("*.html"))
    print(f"\n  Generated {len(html_files)} HTML files:")
    for f in html_files:
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:40s} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
