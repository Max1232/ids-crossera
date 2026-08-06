"""Phase 9's per-attack-family data path: the one-vs-normal contract and the sidecar's key.

Data-free by construction (see ``tests/README.md``): the family breakdown is arithmetic over a
prediction vector, so it is exercised on hand-written labels whose F1 can be computed by hand in the
test itself rather than borrowed from the function under test.

Three things are worth pinning here, and each has a specific failure it prevents:

* **one-vs-normal is what the numbers mean.** An attack family is all-positive by construction, so
  scoring it on its own rows would make "F1" a relabelling of recall and would put it out of reach of
  the majority-class floor every other F1 in this project is read against. If that definition ever
  drifts, every per-family caption becomes wrong without a single test failing.
* **a family and the binary label may never disagree.** ``schema_map`` cross-tabbed them at build
  time; this is the check on the actual evaluation rows, and it is the failure that would silently
  attribute one family's predictions to another.
* **``family_set`` is part of the sidecar's key.** ``dos``, ``scanning`` and ``backdoor`` exist in
  *both* vocabularies over completely different row populations (583 UNSW-test rows vs 20,000
  TON_IoT ones), so without it the cross-era rows and the recovery rows would overwrite each other.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import RANDOM_SEED
from src.evaluate import (
    NATIVE_FAMILY_SET,
    NORMAL_FAMILY,
    PER_FAMILY_HEADER,
    SHARED_FAMILY_SET,
    per_family_metrics,
    per_family_rows,
    read_per_family_metrics,
    write_per_family_metrics,
)


@pytest.fixture
def hand_counted() -> dict[str, np.ndarray]:
    """Six normal rows and two attack families, with every prediction chosen by hand.

    Laid out so the one-vs-normal counts are countable off the page:

    ==========  ======  ======  ====================================
    rows        family  label   prediction
    ==========  ======  ======  ====================================
    0-5         normal  0       attack on rows 0 and 1 (2 FP total)
    6-9         dos     1       attack on rows 6, 7, 8 (3 TP, 1 FN)
    10-12       xss     1       attack on row 10 only  (1 TP, 2 FN)
    ==========  ======  ======  ====================================

    Both families see the *same* six normal rows and therefore the same two false positives, which
    is the property one-vs-normal has and per-family-rows-only scoring does not.
    """
    families = np.array([NORMAL_FAMILY] * 6 + ["dos"] * 4 + ["xss"] * 3, dtype=object)
    y_true = np.array([0] * 6 + [1] * 7)
    y_pred = np.array([1, 1, 0, 0, 0, 0] + [1, 1, 1, 0] + [1, 0, 0])
    # Scores only have to rank; the AUC is not what this fixture is for.
    scores = np.where(y_pred == 1, 0.9, 0.1)
    return {"families": families, "y_true": y_true, "y_pred": y_pred, "scores": scores}


def test_families_are_scored_one_vs_normal(hand_counted: dict[str, np.ndarray]) -> None:
    """Each family's row block plus **every** normal row, with the arithmetic done by hand.

    `dos`: 3 TP, 2 FP (the shared normals), 1 FN -> precision 3/5, recall 3/4, F1 2*0.6*0.75/1.35.
    `xss`: 1 TP, the same 2 FP, 2 FN            -> precision 1/3, recall 1/3, F1 1/3.

    The shared false positives are the point: scoring `xss` on its own three rows would give it no
    false positives at all and a precision of 1.0.
    """
    result = per_family_metrics(**hand_counted)
    assert sorted(result) == ["dos", "xss"]

    dos = result["dos"]
    assert (dos["n_family"], dos["n_normal"], dos["n_test"]) == (4, 6, 10)
    assert dos["precision"] == pytest.approx(3 / 5)
    assert dos["recall"] == pytest.approx(3 / 4)
    assert dos["f1"] == pytest.approx(2 * (3 / 5) * (3 / 4) / ((3 / 5) + (3 / 4)))
    assert dos["positive_rate"] == pytest.approx(4 / 10)

    xss = result["xss"]
    assert (xss["n_family"], xss["n_normal"], xss["n_test"]) == (3, 6, 9)
    assert xss["precision"] == pytest.approx(1 / 3)
    assert xss["recall"] == pytest.approx(1 / 3)
    assert xss["f1"] == pytest.approx(1 / 3)


def test_unmapped_rows_are_excluded_not_pooled(hand_counted: dict[str, np.ndarray]) -> None:
    """A missing family label drops the row from every subset — the shared map's NA rows.

    Under ``family_set="shared"`` this is 101,043 of TON_IoT's 211,043 rows (every attack level with
    no 2015 counterpart). Pooling them into an "other" bucket, or letting them ride along as
    normals, would silently redefine what "restricted to the shared families" means.
    """
    families = hand_counted["families"].copy()
    families[hand_counted["families"] == "xss"] = None
    result = per_family_metrics(
        hand_counted["y_true"], hand_counted["y_pred"], hand_counted["scores"], families
    )
    assert sorted(result) == ["dos"]
    assert result["dos"]["n_test"] == 10  # unchanged: the dropped rows were never in dos's subset


def test_pandas_na_is_treated_as_missing(hand_counted: dict[str, np.ndarray]) -> None:
    """``schema_map`` emits the pandas ``string`` dtype, whose NA is ``pd.NA`` and not ``None``.

    ``pd.NA == "dos"`` raises rather than returning False, so the vector has to be materialized
    through ``to_numpy(na_value=None)``. This is the test for that, not a style preference.
    """
    families = pd.Series(hand_counted["families"], dtype="string")
    families[families == "xss"] = pd.NA
    result = per_family_metrics(
        hand_counted["y_true"], hand_counted["y_pred"], hand_counted["scores"], families
    )
    assert sorted(result) == ["dos"]


def test_family_disagreeing_with_the_binary_label_raises(
    hand_counted: dict[str, np.ndarray]
) -> None:
    """A `normal` row carrying label 1 (or a family row carrying 0) is a data-integrity failure."""
    y_true = hand_counted["y_true"].copy()
    y_true[0] = 1  # a `normal` row that claims to be an attack
    with pytest.raises(ValueError, match="family column and the binary label disagree"):
        per_family_metrics(
            y_true, hand_counted["y_pred"], hand_counted["scores"], hand_counted["families"]
        )

    y_true = hand_counted["y_true"].copy()
    y_true[6] = 0  # a `dos` row that claims to be normal
    with pytest.raises(ValueError, match="family column and the binary label disagree"):
        per_family_metrics(
            y_true, hand_counted["y_pred"], hand_counted["scores"], hand_counted["families"]
        )


def test_a_vocabulary_without_normal_rows_raises(hand_counted: dict[str, np.ndarray]) -> None:
    """No benign rows means no one-vs-normal subset; refuse rather than return degenerate scores."""
    families = np.where(
        hand_counted["families"] == NORMAL_FAMILY, None, hand_counted["families"]
    )
    with pytest.raises(ValueError, match="no `normal` rows"):
        per_family_metrics(
            hand_counted["y_true"], hand_counted["y_pred"], hand_counted["scores"], families
        )


def test_misaligned_family_vector_raises(hand_counted: dict[str, np.ndarray]) -> None:
    """Length mismatch is caught before anything is scored against the wrong rows."""
    with pytest.raises(ValueError, match="a per-family breakdown would be scored"):
        per_family_metrics(
            hand_counted["y_true"], hand_counted["y_pred"], hand_counted["scores"],
            hand_counted["families"][:-1],
        )


def _row(**overrides: object) -> dict[str, object]:
    row = {
        "run_id": "phase6-crossera", "model": "random_forest", "regime": "cross_era",
        "family_set": SHARED_FAMILY_SET, "family": "dos", "seed": RANDOM_SEED,
        "n_family": 20000, "n_normal": 50000, "n_test": 70000, "positive_rate": 0.285714,
        "precision": 0.1, "recall": 0.2, "f1": 0.3, "roc_auc": 0.4,
        "accuracy": 0.5, "balanced_accuracy": 0.6, "macro_f1": 0.7, "notes": "",
    }
    row.update(overrides)
    return row


def test_family_set_is_part_of_the_key(tmp_path) -> None:
    """The same (run_id, model, regime, family) in two vocabularies must coexist, not overwrite.

    ``dos``, ``scanning`` and ``backdoor`` are spelled identically under ``shared`` and ``native``
    but are measured over different row populations. Dropping ``family_set`` from the key would
    silently replace one with the other, and the losing figure would simply be drawn from the wrong
    one.
    """
    path = tmp_path / "per_family_metrics.csv"
    write_per_family_metrics(
        [_row(), _row(family_set=NATIVE_FAMILY_SET, n_family=9893, f1=0.99)], path
    )
    rows = read_per_family_metrics(path)
    assert len(rows) == 2
    keys = {key[3] for key in rows}
    assert keys == {SHARED_FAMILY_SET, NATIVE_FAMILY_SET}


def test_write_upserts_and_is_byte_identical_on_a_re_run(tmp_path) -> None:
    """Same contract as ``log_metrics``: two phases write into one file and a re-run is a no-op diff.

    Phase 6 writes the shared block and Phase 7 the native block; running ``./run.sh`` twice must
    leave the committed file unchanged, or "did this reproduce?" stops being a `git diff`.
    """
    path = tmp_path / "per_family_metrics.csv"
    phase6 = [_row(family="dos"), _row(family="scanning")]
    phase7 = [_row(run_id="phase7-recovery-f0.01", regime="target_frozen_test",
                   family_set=NATIVE_FAMILY_SET, family="ransomware")]

    write_per_family_metrics(phase6, path)
    write_per_family_metrics(phase7, path)
    first = path.read_bytes()
    assert len(read_per_family_metrics(path)) == 3  # Phase 7 landed beside Phase 6, not on it

    write_per_family_metrics(phase6, path)
    write_per_family_metrics(phase7, path)
    assert path.read_bytes() == first

    # And an upsert replaces rather than appends.
    write_per_family_metrics([_row(family="dos", f1=0.123456)], path)
    assert len(read_per_family_metrics(path)) == 3
    assert read_per_family_metrics(path)[
        ("phase6-crossera", "random_forest", "cross_era", SHARED_FAMILY_SET, "dos")
    ]["f1"] == "0.123456"


def test_a_stale_header_raises_rather_than_being_reconciled(tmp_path) -> None:
    """The per-family header is frozen for the same reason ``METRICS_HEADER`` is."""
    path = tmp_path / "per_family_metrics.csv"
    path.write_text("run_id,model,family\nphase6-crossera,random_forest,dos\n")
    with pytest.raises(ValueError, match="per-family header is frozen"):
        read_per_family_metrics(path)


def test_rows_are_rounded_and_carry_the_seed(hand_counted: dict[str, np.ndarray]) -> None:
    """``per_family_rows`` is the one place the two producers agree on shape, seed and rounding."""
    scores = {"per_family": per_family_metrics(**hand_counted)}
    rows = per_family_rows(
        scores, run_id="phase6-crossera", model="random_forest", regime="cross_era",
        family_set=SHARED_FAMILY_SET, note="unit test",
    )
    assert [row["family"] for row in rows] == ["dos", "xss"]  # sorted, deterministically
    assert all(set(row) == set(PER_FAMILY_HEADER) for row in rows)
    assert all(row["seed"] == RANDOM_SEED for row in rows)
    assert rows[1]["f1"] == round(1 / 3, 6)
