"""CPU self-test of the corpus bookkeeping in symbolic/corpus.py (the end-to-end check runs in the merge stage)."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import numpy as np

from symbolic import corpus


def test_task_slices_cover_every_event_once():
    for n, tasks in ((1000, 2), (464008, 32), (7, 3)):
        spans = [corpus.task_slice(n, t, tasks) for t in range(tasks)]
        assert spans[0][0] == 0 and spans[-1][1] == n and all(spans[i][1] == spans[i + 1][0] for i in range(tasks - 1))


def test_source_rows_are_species_major():
    rows = corpus.source_rows(per=10, n=3)
    assert rows.tolist() == [0, 1, 2, 10, 11, 12, 20, 21, 22]


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as exc:
                failed += 1
                print(f"FAIL  {name}: {exc!r}")
    sys.exit(1 if failed else 0)
