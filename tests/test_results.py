import csv

from results import FIELDS, write_results


def test_result_header_and_rows_have_same_column_count(tmp_path):
    row = {field: index for index, field in enumerate(FIELDS)}
    path = tmp_path / "result.tsv"
    write_results(path, [row])
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    assert len(rows) == 2
    assert len(rows[0]) == len(rows[1]) == len(FIELDS)
