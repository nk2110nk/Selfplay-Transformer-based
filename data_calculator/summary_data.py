#!/usr/bin/env python3
"""Summarize negotiation results under data/.

The script creates one table per split (expert/general) and metric.  Each table
has opponent-pair rows, domain columns, and an Average row/column like the
reference table in the prompt.
"""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path


DEFAULT_SPLITS = ("expert", "general")
METRICS = ("my_util", "social", "nash", "step", "agreement")
DERIVED_METRICS = ("agreement",)
DOMAIN_ORDER = (
    "Laptop",
    "ItexvsCypress",
    "IS_BT_Acquisition",
    "Grocery",
    "thompson",
    "Car",
    "EnergySmall_A",
)
AGENT_ORDER = ("Boulware", "Conceder", "Linear", "Atlas3")
EXPECTED_CASES = ("case1", "case2", "case3")
EXPECTED_ROWS_PER_CASE = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize expert/general TSV results into tables and CSV files."
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
        help="Data splits to summarize. Default: expert general",
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
        default=None,
        help="Optional domain names to include. Default: known domains, then extras.",
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


def ordered_names(names: list[str], preferred_order: tuple[str, ...]) -> list[str]:
    known = [name for name in preferred_order if name in names]
    extras = sorted(name for name in names if name not in preferred_order)
    return known + extras


def pair_sort_key(pair: str) -> tuple[int, int, str]:
    left, sep, right = pair.partition("-")
    if not sep:
        return (len(AGENT_ORDER), len(AGENT_ORDER), pair)

    left_index = AGENT_ORDER.index(left) if left in AGENT_ORDER else len(AGENT_ORDER)
    right_index = AGENT_ORDER.index(right) if right in AGENT_ORDER else len(AGENT_ORDER)
    return (left_index, right_index, pair)


def discover_pairs(split_dir: Path, requested_pairs: list[str] | None) -> list[str]:
    if requested_pairs is not None:
        return requested_pairs

    pairs = [path.name for path in split_dir.iterdir() if path.is_dir()]
    return sorted(pairs, key=pair_sort_key)


def discover_domains(pair_dirs: list[Path], requested_domains: list[str] | None) -> list[str]:
    if requested_domains is not None:
        return requested_domains

    domain_names: set[str] = set()
    for pair_dir in pair_dirs:
        if not pair_dir.exists():
            continue
        for domain_dir in pair_dir.iterdir():
            if domain_dir.is_dir() and domain_dir.name in DOMAIN_ORDER:
                domain_names.add(domain_dir.name)
    return ordered_names(list(domain_names), DOMAIN_ORDER)


def case_sort_key(case_dir: Path) -> tuple[int, str]:
    name = case_dir.name
    if name.startswith("case") and name[4:].isdigit():
        return (int(name[4:]), name)
    return (10**9, name)


def read_pair_domain(
    split: str,
    pair: str,
    domain: str,
    domain_dir: Path,
    metric: str,
    strict: bool,
) -> float | None:
    if not domain_dir.exists():
        warn_or_raise(f"missing directory: {domain_dir}", strict)
        return None

    case_dirs = sorted(
        [path for path in domain_dir.iterdir() if path.is_dir() and path.name.startswith("case")],
        key=case_sort_key,
    )
    case_names = tuple(path.name for path in case_dirs)
    if case_names != EXPECTED_CASES:
        warn_or_raise(
            f"{split}/{pair}/{domain}: expected {EXPECTED_CASES}, found {case_names}",
            strict,
        )

    metric_column = metric
    values: list[float] = []
    total_rows = 0
    for case_dir in case_dirs:
        tsv_files = sorted(case_dir.glob("*.tsv"))
        if len(tsv_files) != 1:
            warn_or_raise(
                f"{split}/{pair}/{domain}/{case_dir.name}: expected 1 TSV, found {len(tsv_files)}",
                strict,
            )
        for tsv_file in tsv_files:
            with tsv_file.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                if reader.fieldnames is None or metric_column not in reader.fieldnames:
                    warn_or_raise(
                        f"{tsv_file}: missing metric column {metric_column!r}",
                        strict,
                    )
                    continue

                row_count = 0
                for row in reader:
                    row_count += 1
                    if metric == "agreement":
                        text = str(row[metric_column]).strip()
                        values.append(0.0 if text in {"", "None", "False", "0", "{}", "[]"} else 1.0)
                    else:
                        try:
                            values.append(float(row[metric_column]))
                        except (TypeError, ValueError):
                            warn_or_raise(
                                f"{tsv_file}: non-numeric {metric_column!r} "
                                f"value {row.get(metric_column)!r}",
                                strict,
                            )
                total_rows += row_count

    if not values:
        return None

    expected_rows = len(EXPECTED_CASES) * EXPECTED_ROWS_PER_CASE
    if total_rows != expected_rows:
        warn_or_raise(
            f"{split}/{pair}/{domain}: expected {expected_rows} rows, found {total_rows}",
            strict,
        )

    if metric == "agreement":
        return mean(values) * 100.0
    return mean(values)


def warn_or_raise(message: str, strict: bool) -> None:
    if strict:
        raise ValueError(message)
    warnings.warn(message, stacklevel=2)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def build_table(
    data_dir: Path,
    split: str,
    metric: str,
    pairs: list[str],
    domains: list[str],
    strict: bool,
    decimals: int,
) -> list[list[str]]:
    split_dir = data_dir / split
    is_percent_metric = metric in DERIVED_METRICS
    raw_table: dict[str, dict[str, float | None]] = {
        pair: {domain: None for domain in domains} for pair in pairs
    }

    for pair in pairs:
        for domain in domains:
            mean_value = read_pair_domain(
                split=split,
                pair=pair,
                domain=domain,
                domain_dir=split_dir / pair / domain,
                metric=metric,
                strict=strict,
            )
            if mean_value is not None:
                raw_table[pair][domain] = mean_value

    rows: list[list[str]] = [[""] + domains + ["Average"]]
    row_averages: list[float] = []

    for pair in pairs:
        domain_values = [raw_table[pair][domain] for domain in domains]
        present_values = [value for value in domain_values if value is not None]
        row_average = mean(present_values) if present_values else None
        if row_average is not None:
            row_averages.append(row_average)
        rows.append(
            [pair]
            + [
                format_number(value, decimals, percent=is_percent_metric)
                for value in domain_values
            ]
            + [format_number(row_average, decimals, percent=is_percent_metric)]
        )

    average_row = ["Average"]
    for domain in domains:
        column_values = [
            raw_table[pair][domain] for pair in pairs if raw_table[pair][domain] is not None
        ]
        average_row.append(
            format_number(
                mean(column_values) if column_values else None,
                decimals,
                percent=is_percent_metric,
            )
        )
    average_row.append(
        format_number(
            mean(row_averages) if row_averages else None,
            decimals,
            percent=is_percent_metric,
        )
    )
    rows.append(average_row)
    return rows


def format_number(value: float | None, decimals: int, percent: bool = False) -> str:
    if value is None:
        return ""
    suffix = "%" if percent else ""
    return f"{value:.{decimals}f}{suffix}"


def render_table(rows: list[list[str]]) -> str:
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    rendered_lines = []
    for row in rows:
        rendered_lines.append(
            "  ".join(value.rjust(widths[index]) for index, value in enumerate(row)).rstrip()
        )
    return "\n".join(rendered_lines)


def write_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rendered_tables: list[str] = []

    for split in args.splits:
        split_dir = args.data_dir / split
        if not split_dir.exists():
            warn_or_raise(f"missing split directory: {split_dir}", args.strict)
            continue

        pairs = discover_pairs(split_dir, args.pairs)
        pair_dirs = [split_dir / pair for pair in pairs]
        domains = discover_domains(pair_dirs, args.domains)

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

            csv_path = args.output_dir / f"{split}_{metric}.csv"
            write_csv(csv_path, table)

    table_text_path = args.output_dir / "tables.txt"
    table_text_path.write_text("\n\n".join(rendered_tables) + "\n", encoding="utf-8")
    print(f"\nSaved CSV files and table text to: {args.output_dir}")


if __name__ == "__main__":
    main()
