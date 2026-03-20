"""Run all data collectors and save combined results.

Usage:
    python -m west_africa.scripts.collect_all
    python -m west_africa.scripts.collect_all --start-year 2010 --end-year 2024
    python -m west_africa.scripts.collect_all --comtrade-key YOUR_API_KEY
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys

# Ensure project root is on sys.path
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from west_africa.collectors.scheduler import CollectionScheduler
from west_africa.core.graph import DATA_DIR

CACHE_DIR = DATA_DIR / ".cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all West Africa FTZ data collectors."
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2000,
        help="Start year for data collection (default: 2000)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
        help="End year for data collection (default: 2025)",
    )
    parser.add_argument(
        "--comtrade-key",
        type=str,
        default="",
        help="UN Comtrade API key (optional, some endpoints work without it)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()

    print("West Africa FTZ -- Data Collection")
    print(f"  Period: {args.start_year} - {args.end_year}")
    print(f"  Comtrade key: {'provided' if args.comtrade_key else 'not set'}")
    print()

    scheduler = CollectionScheduler(comtrade_api_key=args.comtrade_key)

    print("Starting collection from all sources...")
    print("-" * 50)
    results = scheduler.collect_all(
        start_year=args.start_year,
        end_year=args.end_year,
    )

    # Print summary
    print()
    print("=" * 50)
    print("  COLLECTION RESULTS SUMMARY")
    print("=" * 50)
    print(f"  {'Source':20s} {'Records':>10s}")
    print(f"  {'-' * 20} {'-' * 10}")

    total = 0
    for source, records in sorted(results.items()):
        count = len(records)
        total += count
        status = "OK" if count > 0 else "EMPTY"
        print(f"  {source:20s} {count:10d}  [{status}]")

    print(f"  {'-' * 20} {'-' * 10}")
    print(f"  {'TOTAL':20s} {total:10d}")

    # Save to cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CACHE_DIR / "collection_results.json"

    # Convert results to JSON-serializable format
    serializable = {}
    for source, records in results.items():
        serializable[source] = records

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    if total == 0:
        print("\nWARNING: No records collected from any source.")
        print("Check network connectivity and API keys.")


if __name__ == "__main__":
    main()
