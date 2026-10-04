#!/usr/bin/env python3
"""Summarize unexpected-domain negotiation results under data/.

This is the Coffee/Camera/Lunch/SmartPhone/Kitchen counterpart to
summary_data.py. It keeps the same metrics and table format, but writes files
with an unexpect_ prefix so the original summary tables are left separate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from summary_data import (
    METRICS,
    build_table,
    discover_pairs,
    render_table,
    warn_or_raise,
    write_csv,
)


DEFAULT_SPLITS = ("general",)
UNEXPECT_DOMAINS = (
    "Coffee",
    "Camera",
    "Lunch",
    "SmartPhone",
    "Kitchen",
)
OUTPUT_PREFIX = "unexpect"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize Coffee/Camera/Lunch/SmartPhone/Kitchen TSV results "
            "into tables and CSV files."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root data directory. Default: data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("summary_tables"),
        help="Directory for CSV and text table outputs. Default: summary_tables",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Data splits to summarize. Default: general",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(METRICS),
        help="Metric columns to summarize. Default: my_util social nash step agreement",
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        help="Optional pair names to include, e.g. Boulware-Boulware Conceder-Linear.",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=list(UNEXPECT_DOMAINS),
        help=(
            "Domain names to include. Default: Coffee Camera Lunch "
            "SmartPhone Kitchen"
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when a pair/domain does not have exactly case1-case3 and 300 rows.",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=3,
        help="Number of decimals for output tables. Default: 3",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    domains = args.domains
    rendered_tables: list[str] = []

    for split in args.splits:
        split_dir = args.data_dir / split
        if not split_dir.exists():
            warn_or_raise(f"missing split directory: {split_dir}", args.strict)
            continue

        pairs = discover_pairs(split_dir, args.pairs)

        for metric in args.metrics:
            table = build_table(
                data_dir=args.data_dir,
                split=split,
                metric=metric,
                pairs=pairs,
                domains=domains,
                strict=args.strict,
                decimals=args.round,
            )

            title = f"{split} / {metric}"
            rendered = f"\n=== {title} ===\n{render_table(table)}"
            print(rendered)
            rendered_tables.append(rendered.strip())

            csv_path = args.output_dir / f"{OUTPUT_PREFIX}_{split}_{metric}.csv"
            write_csv(csv_path, table)

    table_text_path = args.output_dir / f"{OUTPUT_PREFIX}_tables.txt"
    table_text_path.write_text("\n\n".join(rendered_tables) + "\n", encoding="utf-8")
    print(f"\nSaved CSV files and table text to: {args.output_dir}")


if __name__ == "__main__":
    main()
