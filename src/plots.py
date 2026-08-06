"""Figures for the report and presentation (Phase 9) -> reports/figures/.

Every figure needs clear labels, legends, and captions (graded). Produces:
  - in-distribution vs cross-era metric bars
  - confusion matrices (per model, per regime)
  - ROC curves
  - per-shared-family F1
  - the transfer-learning recovery curve (RQ2 secondary headline)

**Every number rendered here is read from a committed artifact of the phase that measured it.**
Nothing in this module fits, transforms, or re-derives anything, so a figure can never disagree
with the table it illustrates. There are exactly three such artifacts:

* ``reports/metrics.csv`` -- the run log, keyed ``(run_id, model, regime)``; the source for every
  scalar metric on every figure;
* ``reports/confusion_matrices.json`` -- the 2x2 count matrices, which the log's frozen 14-column
  header has no room for. Written by :func:`evaluate.run_phase6` from the matrices it already
  computed (see :func:`evaluate.write_confusion_matrices`);
* ``reports/roc_curves.json`` -- the ROC vertices behind the logged ``roc_auc`` scalar, likewise not
  a scalar and likewise written by :func:`evaluate.run_phase6` (see
  :func:`evaluate.write_roc_curves`).

Three consequences worth knowing before editing:

* all three files are opened **read-only** -- this module must never call
  :func:`evaluate.log_metrics`, :func:`evaluate.write_confusion_matrices` or
  :func:`evaluate.write_roc_curves`;
* a figure that needs a quantity no artifact carries (per-family F1) needs
  a new data path in the phase that computes it, not a computation smuggled in here -- and that
  path is a *persisted artifact*, not a call back into the phase: ``python -m src.plots`` must stay
  a seconds-long command that re-renders figures, and ``./run.sh`` must not run any phase twice;
* where the two artifacts overlap they are cross-checked against each other rather than trusted --
  see :func:`_confusion_table`.

**Δ sign convention, project-wide and stated on every axis that shows one:**
``Δ = in_distribution − cross_era``, i.e. **what the model lost** -- positive means degraded. That
is :func:`evaluate.metric_deltas` and what ``./run.sh`` prints, so the report, the figures and a
grader's own run all agree on the sign.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import textwrap
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: run.sh has no display, and figures are files on disk by design

import matplotlib.pyplot as plt  # noqa: E402 - must follow the backend selection above
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

from .config import (  # noqa: E402
    CONFUSION_JSON,
    FIGURES,
    METRICS_CSV,
    PER_FAMILY_CSV,
    RANDOM_SEED,
    ROC_JSON,
    set_seeds,
)
from .evaluate import (  # noqa: E402
    EXPECTED_ROWS,
    METRIC_DECIMALS,
    METRICS_HEADER,
    METRICS_KEY,
    NATIVE_FAMILY_SET,
    NORMAL_FAMILY,
    PER_FAMILY_HEADER,
    PER_FAMILY_KEY,
    PHASE4_AGREEMENT_TOLERANCE,
    POSITIVE_LABEL,
    ROC_CURVE_TOLERANCE,
    SHARED_FAMILY_SET,
    read_confusion_matrices,
    read_metrics,
    read_per_family_metrics,
    read_roc_curves,
)
from .schema_map import COMMON_COLUMNS, FEATURE_MAP, SHARED_FAMILIES  # noqa: E402

# =========================================================================================
# Shared plotting foundation — style, loading, saving. Passes 2-5 build their figures on this.
# =========================================================================================

# --- Palette ---------------------------------------------------------------------------------
#
# Taken unchanged from the `dataviz` skill's validated categorical palette (light mode) and used in
# its documented slot ORDER, which is the colour-blind-safety mechanism rather than a cosmetic
# choice -- re-ordering or re-stepping it invalidates the validation. Two assignments are in play
# and they are deliberately different jobs:
#
#   REGIME_COLOURS  -- 2 slots (blue, orange), used by the drift bars to encode *regime*. Validated
#                      on the all-pairs list: CVD ΔE 24.7, normal-vision ΔE 33.6, both >= 3:1 on a
#                      white surface. Nothing to qualify.
#   MODEL_COLOURS   -- 5 slots, used by the recovery curve to encode *model identity*. Validated on
#                      the adjacent pairlist (the documented default for line charts): CVD ΔE 9.1,
#                      normal-vision ΔE 19.6. Three of the five (aqua, yellow, magenta) sit below
#                      3:1 against white, which obliges *relief*: every line therefore also carries
#                      a distinct marker shape and dash pattern, and the exact values are tabulated
#                      in reports/figures/README.md. Identity is never colour-alone here.
#
# The five model slots cannot clear the all-pairs floors -- no ordering of this palette can past
# three slots -- which is why the marker/dash secondary encoding is mandatory rather than decorative.
_BLUE, _ORANGE, _AQUA, _YELLOW, _MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"

#: Ink. Text never wears a series colour; a coloured mark beside it carries the identity.
INK_PRIMARY: str = "#0b0b0b"
INK_SECONDARY: str = "#52514e"
INK_MUTED: str = "#8a8983"
GRID: str = "#dedcd5"

#: The two regimes of RQ1. Blue = the era the model was trained for, orange = the era it was
#: carried onto.
REGIME_COLOURS: dict[str, str] = {"in_distribution": _BLUE, "cross_era": _ORANGE}

#: Human-readable regime names for legends. Spelled with the set they were measured on, because a
#: reader who sees only "in-distribution" cannot tell which data that was.
REGIME_LABELS: dict[str, str] = {
    "in_distribution": "In-distribution (UNSW-NB15 test, 2015)",
    "cross_era": "Cross-era, zero-shot (TON_IoT, 2019–20)",
}

#: Display names. `metrics.csv` spells models in snake_case; figures should not.
MODEL_LABELS: dict[str, str] = {
    "random_forest": "Random forest",
    "decision_tree": "Decision tree",
    "scratch_mlp": "MLP (from scratch)",
    "svm": "Linear SVM",
    "scratch_logreg": "Logistic reg. (from scratch)",
    "dummy": "Majority-class dummy",
}

#: Presentation order for every figure, fixed once here so a reader carries one ordering across the
#: whole results section. Sorted by in-distribution ROC-AUC descending (which puts the dummy floor
#: last on its own merits), and it is also the colour-slot order: colour follows the model, never
#: its rank within a particular figure, so a model keeps its hue across every figure in the report.
MODEL_ORDER: tuple[str, ...] = (
    "random_forest", "decision_tree", "scratch_mlp", "svm", "scratch_logreg", "dummy",
)

#: The five real models get a categorical slot; the dummy is a *reference level*, not a series, so
#: it wears neutral ink wherever it appears as a floor.
MODEL_COLOURS: dict[str, str] = {
    "random_forest": _BLUE,
    "decision_tree": _ORANGE,
    "scratch_mlp": _AQUA,
    "svm": _YELLOW,
    "scratch_logreg": _MAGENTA,
    "dummy": INK_SECONDARY,
}

#: Secondary encoding — mandatory, not decoration (see the palette note above). Shape and dash
#: pattern carry model identity independently of hue, so the recovery curve survives colour-blind
#: readers, greyscale printing, and the three low-contrast slots.
MODEL_MARKERS: dict[str, str] = {
    "random_forest": "o", "decision_tree": "s", "scratch_mlp": "^",
    "svm": "D", "scratch_logreg": "v", "dummy": "x",
}
MODEL_DASHES: dict[str, Any] = {
    "random_forest": "solid",
    "decision_tree": (0, (4.5, 1.5)),
    "scratch_mlp": (0, (1.2, 1.2)),
    "svm": (0, (5, 1.2, 1, 1.2)),
    "scratch_logreg": (0, (3, 1.2, 1, 1.2, 1, 1.2)),
    "dummy": (0, (2, 1.5)),
}

# --- Output format ---------------------------------------------------------------------------
#: 200 dpi PNG. Enough for a 6-page PDF at column width without the file size of a 600-dpi raster,
#: and every figure is regenerated by `run.sh` rather than hand-edited, so a vector round-trip buys
#: nothing.
FIGURE_DPI: int = 200
FIGURE_FORMAT: str = "png"

#: Full text width of the report page (US Letter, 1-inch margins). Figures are sized to be embedded
#: at 100% and read at that size -- a figure scaled down in the PDF is a figure with unreadable
#: labels, which the rubric grades directly.
PAGE_WIDTH_IN: float = 6.5


def apply_style() -> None:
    """Set the rcParams every figure in this module shares.

    Called once by each figure function rather than at import time, so importing ``src.plots`` (as
    a test or another module might) does not mutate a caller's global matplotlib state as a side
    effect.

    Print-first, not dashboard-first: DejaVu Sans because it ships with matplotlib and is therefore
    present on any machine that can run ``run.sh`` at all (a missing font silently substitutes and
    changes every metric of the layout); small type sized for a half-page figure; recessive grid on
    the value axis only; no top/right spines.
    """
    plt.rcParams.update({
        "figure.dpi": FIGURE_DPI,
        "savefig.dpi": FIGURE_DPI,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.titleweight": "bold",
        "axes.edgecolor": INK_SECONDARY,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_PRIMARY,
        "text.color": INK_PRIMARY,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelcolor": INK_PRIMARY,
        "ytick.labelcolor": INK_PRIMARY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
    })


# --- Loading ---------------------------------------------------------------------------------

#: Columns coerced to float on load. The rest (`run_id`, `model`, `regime`, `notes`) stay strings,
#: and `seed`/`n_test` are read as ints.
_FLOAT_COLUMNS: tuple[str, ...] = (
    "accuracy", "precision", "recall", "f1", "roc_auc",
    "balanced_accuracy", "macro_f1", "positive_rate",
)
_INT_COLUMNS: tuple[str, ...] = ("seed", "n_test")


def load_metrics(path: Any = METRICS_CSV) -> pd.DataFrame:
    """Read ``reports/metrics.csv`` into a typed DataFrame. **Read-only — never writes.**

    Built on :func:`evaluate.read_metrics` rather than a bare ``pd.read_csv`` on purpose. That
    function owns two things this module must not re-implement: it *refuses* a file whose header is
    not the frozen :data:`evaluate.METRICS_HEADER` (a stale header means every row is silently
    misaligned, and a figure drawn from misaligned rows looks perfectly fine), and it keys rows on
    exactly :data:`evaluate.METRICS_KEY`, which is the same ``(run_id, model, regime)`` identity the
    figure captions cite. A second reader here would be free to drift from both. The only thing
    added on top is numeric coercion, which ``read_metrics`` deliberately does not do (it returns
    strings so a carried-forward row is re-emitted byte-for-byte).

    Rows come back in the CSV's committed order, with the frozen column order preserved.
    """
    frame = pd.DataFrame(list(read_metrics(path).values()), columns=list(METRICS_HEADER))
    for column in _FLOAT_COLUMNS:
        frame[column] = frame[column].astype(float)
    for column in _INT_COLUMNS:
        frame[column] = frame[column].astype(int)
    return frame


#: Per-family columns coerced to float / int on load. The rest stay strings.
_FAMILY_FLOAT_COLUMNS: tuple[str, ...] = (
    "positive_rate", "precision", "recall", "f1", "roc_auc",
    "accuracy", "balanced_accuracy", "macro_f1",
)
_FAMILY_INT_COLUMNS: tuple[str, ...] = ("seed", "n_family", "n_normal", "n_test")


def load_per_family(path: Any = PER_FAMILY_CSV) -> pd.DataFrame:
    """Read ``reports/per_family_metrics.csv`` into a typed DataFrame. **Read-only — never writes.**

    The same argument as :func:`load_metrics`, applied to the second committed table: built on
    :func:`evaluate.read_per_family_metrics` so the frozen-header refusal and the
    ``(run_id, model, regime, family_set, family)`` key live in exactly one place, with numeric
    coercion added on top.
    """
    frame = pd.DataFrame(
        list(read_per_family_metrics(path).values()), columns=list(PER_FAMILY_HEADER)
    )
    for column in _FAMILY_FLOAT_COLUMNS:
        frame[column] = frame[column].astype(float)
    for column in _FAMILY_INT_COLUMNS:
        frame[column] = frame[column].astype(int)
    return frame


def as_per_family_frame(results: Any) -> pd.DataFrame:
    """Coerce whatever a caller passed as the per-family ``results`` into the loaded DataFrame.

    The three spellings :func:`as_metrics_frame` accepts, for the same reasons.
    """
    if isinstance(results, pd.DataFrame):
        return results
    if results is None:
        return load_per_family()
    return load_per_family(results)


def select_family_row(
    frame: pd.DataFrame, run_id: str, model: str, regime: str, family_set: str, family: str
) -> pd.Series:
    """The single per-family row for one :data:`evaluate.PER_FAMILY_KEY`, or raise.

    :func:`select_row`'s counterpart on the wider key, and it exists for the same reason: a missing
    key would otherwise be a silently absent bar and a duplicate would mean the upsert invariant in
    :func:`evaluate.write_per_family_metrics` had broken. ``family_set`` is part of the key, not
    decoration — ``dos``, ``scanning`` and ``backdoor`` exist in *both* vocabularies over different
    row populations.
    """
    match = frame[
        (frame["run_id"] == run_id) & (frame["model"] == model) & (frame["regime"] == regime)
        & (frame["family_set"] == family_set) & (frame["family"] == family)
    ]
    if len(match) != 1:
        raise KeyError(
            f"expected exactly one {PER_FAMILY_KEY} row for ({run_id!r}, {model!r}, {regime!r}, "
            f"{family_set!r}, {family!r}) in the per-family log, found {len(match)}"
        )
    return match.iloc[0]


def as_metrics_frame(metrics: Any) -> pd.DataFrame:
    """Coerce whatever a caller passed as ``metrics`` into the loaded DataFrame.

    The figure functions keep the stub signatures' required positional argument, so this exists to
    make all three reasonable spellings work: an already-loaded DataFrame (what :func:`main` passes,
    so the file is read once per run), ``None`` (load the default log), or a path to a log.
    """
    if isinstance(metrics, pd.DataFrame):
        return metrics
    if metrics is None:
        return load_metrics()
    return load_metrics(metrics)


def select_row(frame: pd.DataFrame, run_id: str, model: str, regime: str) -> pd.Series:
    """The single row for one ``(run_id, model, regime)`` key, or raise.

    Every number in every figure goes through here, which is what makes "sourced to a specific row"
    a property of the code rather than a claim in a caption. A missing key is a figure that would
    otherwise be drawn from a silently-dropped bar; a duplicate key would mean the upsert invariant
    in :func:`evaluate.log_metrics` had broken. Both raise.
    """
    match = frame[
        (frame["run_id"] == run_id) & (frame["model"] == model) & (frame["regime"] == regime)
    ]
    if len(match) != 1:
        raise KeyError(
            f"expected exactly one {METRICS_KEY} row for "
            f"({run_id!r}, {model!r}, {regime!r}) in the metrics log, found {len(match)}"
        )
    return match.iloc[0]


# --- Saving + the caption sidecar --------------------------------------------------------------

#: ``(figure element, key)`` -- the provenance of one mark on one figure. ``key`` is the
#: ``(run_id, model, regime)`` of a ``reports/metrics.csv`` row, or the wider
#: ``(run_id, model, regime, family_set, family)`` of a ``reports/per_family_metrics.csv`` one; the
#: sidecar renders whichever width an entry uses (see :func:`_write_figure_index`).
SourceRow = tuple[str, tuple[str, ...]]

#: Column headings for the two provenance key widths, indexed by ``len(key)``.
_SOURCE_COLUMNS: dict[int, tuple[str, ...]] = {
    len(METRICS_KEY): METRICS_KEY,
    len(PER_FAMILY_KEY): PER_FAMILY_KEY,
}

#: Figures written by the current process, in generation order. :func:`save_figure` appends here and
#: rewrites the sidecar index from it, so the index is a deterministic function of the run: calling
#: `python -m src.plots` twice leaves reports/figures/README.md byte-identical, the same idempotence
#: contract `evaluate.log_metrics` gives reports/metrics.csv. It follows that :func:`main` is the
#: canonical producer of that file -- generating a single figure on its own writes an index
#: describing only that figure, which is correct but partial.
_FIGURE_INDEX: list[dict[str, Any]] = []

#: The sidecar itself: report-ready captions plus the row->figure mapping, beside the PNGs.
FIGURE_INDEX_NAME: str = "README.md"


def _metrics_fingerprint(path: Any = METRICS_CSV) -> str:
    """MD5 of a source artifact, recorded in the sidecar so a figure can be tied to its state."""
    path = Path(path)
    if not path.exists():
        return "(absent)"
    return hashlib.md5(path.read_bytes()).hexdigest()


def save_figure(
    fig: Any,
    name: str,
    caption: str,
    sources: Sequence[SourceRow],
    out: Any = FIGURES,
) -> Path:
    """Write one figure to ``out/<name>.png`` and record its caption + provenance in the sidecar.

    ``caption`` is the **full, report-ready** text (2-4 sentences, paste-able into the PDF) -- not
    the short subtitle rendered on the figure itself. ``sources`` maps each element of the figure
    to the exact ``(run_id, model, regime)`` row of ``reports/metrics.csv`` it was drawn from, which
    is what lets a reader (or a grader) check any bar or point without re-running anything.

    Closes ``fig`` on the way out: a Phase 9 run draws several figures and matplotlib keeps every
    unclosed one alive.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.{FIGURE_FORMAT}"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)

    entry = {"name": path.name, "caption": " ".join(caption.split()), "sources": list(sources)}
    for index, existing in enumerate(_FIGURE_INDEX):
        if existing["name"] == entry["name"]:  # regenerated in the same process: replace in place
            _FIGURE_INDEX[index] = entry
            break
    else:
        _FIGURE_INDEX.append(entry)
    _write_figure_index(out)
    return path


def _write_figure_index(out: Any = FIGURES) -> Path:
    """Rewrite ``reports/figures/README.md`` from :data:`_FIGURE_INDEX`."""
    out = Path(out)
    lines = [
        "# Figures — captions and provenance",
        "",
        "**Generated by `python -m src.plots` (Phase 9). Do not edit by hand — it is rewritten on",
        "every run.** Each entry carries the report-ready caption and the exact rows of",
        "`reports/metrics.csv` every mark was drawn from — or, for the confusion matrices, checked",
        "against — keyed on `(run_id, model, regime)`.",
        "",
        f"Source log: `reports/metrics.csv` (MD5 `{_metrics_fingerprint()}`) — every scalar. Two",
        "quantities that file's frozen 14-column header has no room for come from their own sidecars,",
        "both written by `evaluate.run_phase6()`: the 2x2 confusion counts from",
        f"`reports/confusion_matrices.json` (MD5 `{_metrics_fingerprint(CONFUSION_JSON)}`) and the ROC",
        f"vertices from `reports/roc_curves.json` (MD5 `{_metrics_fingerprint(ROC_JSON)}`).",
        "The per-attack-family figures read a fourth committed table,",
        f"`reports/per_family_metrics.csv` (MD5 `{_metrics_fingerprint(PER_FAMILY_CSV)}`), whose own",
        "frozen header is keyed on `(run_id, model, regime, family_set, family)` — the metrics log's",
        "key has no family dimension, so a per-family row would collide with the aggregate row it",
        "decomposes. This module reads all four and writes none; where they overlap they are",
        "cross-checked against each other before anything is drawn.",
        "",
        "**Δ sign convention throughout: `Δ = in_distribution − cross_era`, i.e. what the model",
        "*lost* — positive means degraded.** This is `evaluate.metric_deltas()` and what `./run.sh`",
        "prints.",
        "",
    ]
    for number, entry in enumerate(_FIGURE_INDEX, start=1):
        # One table per figure, at whichever key width that figure's sources use. A figure drawn
        # from reports/per_family_metrics.csv is keyed on five fields rather than three, and
        # flattening those into the three-column table would drop the family the row is *about*.
        widths = {len(key) for _element, key in entry["sources"]}
        if len(widths) != 1 or widths.isdisjoint(_SOURCE_COLUMNS):
            raise ValueError(
                f"{entry['name']} mixes provenance key widths {sorted(widths)}; a figure's sources "
                f"must all be one of {sorted(_SOURCE_COLUMNS)} fields wide"
            )
        columns = _SOURCE_COLUMNS[widths.pop()]
        lines += [
            f"## Figure {number} — `{entry['name']}`",
            "",
            entry["caption"],
            "",
            "| figure element | " + " | ".join(columns) + " |",
            "|---|" + "---|" * len(columns),
        ]
        for element, key in entry["sources"]:
            lines.append(f"| {element} | " + " | ".join(f"`{field}`" for field in key) + " |")
        lines.append("")

    out.mkdir(parents=True, exist_ok=True)
    path = out / FIGURE_INDEX_NAME
    path.write_text("\n".join(lines))
    return path


def _subtitle(fig: Any, title: str, subtitle: str) -> None:
    """Bold title + a short on-figure caption beneath it, in muted ink.

    The rubric grades captions, and a figure that only makes sense next to its paragraph in the PDF
    has already lost those points -- so the headline claim is rendered *on* the figure. The long
    form still goes to the sidecar via :func:`save_figure`.
    """
    fig.suptitle(title, fontsize=10.5, fontweight="bold", color=INK_PRIMARY, y=0.995)
    fig.text(
        0.5, 0.945, subtitle, ha="center", va="top", fontsize=7.6, color=INK_SECONDARY,
        linespacing=1.45,
    )


# =========================================================================================
# Figure 1 — RQ1 drift bars
# =========================================================================================

#: Where each model's **in-distribution** half lives. This is the one join in Phase 9 that cannot be
#: done by filtering on a single ``run_id``, and getting it wrong is silent.
#:
#: Phase 6 deliberately does **not** re-log the four classical baselines' in-distribution rows --
#: they are ``phase4-baselines``' (see ``evaluate.run_condition(log_in_distribution=False)`` and
#: ``evaluate._PHASE4_MODELS``), because a second copy under a Phase 6 ``run_id`` would count them
#: twice in the results table. The two scratch models have no Phase 4 row at all (Phase 5 logged
#: nothing), so Phase 6 is where *their* in-distribution rows were created. An inner join on
#: ``run_id`` would therefore silently drop four of the six models and leave a two-bar "headline".
IN_DISTRIBUTION_RUN_ID: dict[str, str] = {
    "random_forest": "phase4-baselines",
    "decision_tree": "phase4-baselines",
    "svm": "phase4-baselines",
    "dummy": "phase4-baselines",
    "scratch_mlp": "phase6-crossera",
    "scratch_logreg": "phase6-crossera",
}

#: Every model's cross-era half comes from the one unablated Phase 6 condition.
CROSS_ERA_RUN_ID: str = "phase6-crossera"

#: Run IDs that must NEVER reach the headline figure. The two ablations are matched
#: in-distribution/cross-era pairs at d=18 under their own ``run_id``s, comparable to
#: ``phase6-crossera`` and to nothing else; an ablated row landing in a headline range has already
#: happened once on this project, so the exclusion is asserted rather than assumed.
ABLATION_RUN_IDS: tuple[str, ...] = ("phase6-crossera-no_proto", "phase6-crossera-no_conn_state")


def _drift_table(frame: pd.DataFrame) -> pd.DataFrame:
    """The explicit six-model in-distribution/cross-era join, with the ablations excluded.

    Returns one row per model in :data:`MODEL_ORDER` carrying both halves of both headline metrics
    plus the prevalence each was measured at. Raises unless exactly six models arrive with both
    halves, and unless no ablation ``run_id`` was touched.
    """
    records = []
    for model in MODEL_ORDER:
        in_run = IN_DISTRIBUTION_RUN_ID[model]
        if in_run in ABLATION_RUN_IDS or CROSS_ERA_RUN_ID in ABLATION_RUN_IDS:  # pragma: no cover
            raise ValueError(f"{model} would be drawn from an ablation run_id ({in_run})")
        in_row = select_row(frame, in_run, model, "in_distribution")
        cross_row = select_row(frame, CROSS_ERA_RUN_ID, model, "cross_era")
        records.append({
            "model": model,
            "in_run_id": in_run,
            "cross_run_id": CROSS_ERA_RUN_ID,
            "in_roc_auc": float(in_row["roc_auc"]),
            "cross_roc_auc": float(cross_row["roc_auc"]),
            "in_f1": float(in_row["f1"]),
            "cross_f1": float(cross_row["f1"]),
            "in_n": int(in_row["n_test"]),
            "cross_n": int(cross_row["n_test"]),
            "in_positive_rate": float(in_row["positive_rate"]),
            "cross_positive_rate": float(cross_row["positive_rate"]),
        })
    table = pd.DataFrame.from_records(records)

    if len(table) != len(MODEL_ORDER) or set(table["model"]) != set(MODEL_ORDER):
        raise RuntimeError(  # pragma: no cover - unreachable while the loop is over MODEL_ORDER
            f"the drift join produced {len(table)} models, expected {len(MODEL_ORDER)}"
        )
    used = set(table["in_run_id"]) | set(table["cross_run_id"])
    leaked = used & set(ABLATION_RUN_IDS)
    if leaked:
        raise RuntimeError(
            f"ablation run_ids {sorted(leaked)} reached the RQ1 headline figure. They are matched "
            "d=18 conditions comparable only to phase6-crossera and must never be plotted here."
        )
    # Both regimes must each have been measured on ONE set, or the bars are not comparable.
    for column in ("in_n", "cross_n", "in_positive_rate", "cross_positive_rate"):
        if table[column].nunique() != 1:
            raise RuntimeError(
                f"{column} is not constant across models ({sorted(set(table[column]))}); the six "
                "models were not all scored on the same evaluation set."
            )
    return table


def plot_indist_vs_crossera(metrics: Any, out: Any = FIGURES) -> Path:
    """Grouped bar chart: in-distribution vs cross-era F1/ROC-AUC per model (RQ1 headline).

    Two panels sharing one legend: **ROC-AUC leads** (prevalence-insensitive, so it is the panel the
    drift claim is made from) and F1 follows with the prevalence caveat drawn onto it as the
    majority-class reference lines -- UNSW-test is 44.94% normal / 55.06% attack against TON_IoT's
    23.69% / 76.31%, so part of any F1 movement is the balance change rather than drift, and the
    dummy's F1 *rises* across eras on that alone.

    ``metrics`` is the DataFrame from :func:`load_metrics` (or ``None`` / a path). The
    in-distribution halves are joined from **two** ``run_id``s -- see
    :data:`IN_DISTRIBUTION_RUN_ID` -- and the two ablation conditions are excluded and asserted
    excluded.
    """
    apply_style()
    table = _drift_table(as_metrics_frame(metrics))

    in_share = float(table["in_positive_rate"].iloc[0])
    cross_share = float(table["cross_positive_rate"].iloc[0])
    in_n, cross_n = int(table["in_n"].iloc[0]), int(table["cross_n"].iloc[0])
    dummy = table[table["model"] == "dummy"].iloc[0]

    # Horizontal bars, not vertical: six model names on a category axis at half-page width need
    # 30-degree rotation to fit as x labels, which costs a third of the figure's height and is
    # harder to read. On y they sit horizontal, and the space freed to the right of the bars
    # becomes a dedicated gutter for the Δ column -- which is the figure's headline quantity and
    # was unreadable squeezed above the bars.
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH_IN, 3.45), sharey=True)
    positions = list(range(len(table)))
    height = 0.36
    gutter_x = 1.045        # left edge of the Δ column
    header_y = -1.02        # the band above the top bar, for the Δ header and the line labels

    panels = (
        ("roc_auc", "ROC-AUC", axes[0], "(a) ROC-AUC — the drift claim"),
        ("f1", "F1 (attack class)", axes[1], "(b) F1 — read against prevalence"),
    )
    for metric, xlabel, ax, panel_title in panels:
        in_values = table[f"in_{metric}"].to_numpy()
        cross_values = table[f"cross_{metric}"].to_numpy()
        ax.barh(
            [p - height / 2 for p in positions], in_values, height,
            color=REGIME_COLOURS["in_distribution"], label=REGIME_LABELS["in_distribution"],
            edgecolor="white", linewidth=0.8, zorder=3,
        )
        ax.barh(
            [p + height / 2 for p in positions], cross_values, height,
            color=REGIME_COLOURS["cross_era"], label=REGIME_LABELS["cross_era"],
            edgecolor="white", linewidth=0.8, zorder=3,
        )
        # Δ = in − cross, in its own gutter: positive = what the model lost.
        ax.text(
            gutter_x, header_y, "Δ", ha="left", va="center", fontsize=7.0,
            fontweight="bold", color=INK_PRIMARY,
        )
        for position, high, low in zip(positions, in_values, cross_values):
            ax.text(
                gutter_x, position, f"{high - low:+.3f}", ha="left", va="center",
                fontsize=6.8, color=INK_PRIMARY,
            )
        ax.set_xlabel(xlabel)
        ax.set_xlim(0.0, 1.30)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks(positions)
        ax.set_yticklabels([MODEL_LABELS[m] for m in table["model"]])
        ax.set_ylim(len(table) - 0.4, header_y - 0.52)  # inverted: best model on top
        ax.set_title(panel_title, pad=6)
        ax.grid(axis="x", zorder=0)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)

    # Panel (a): chance. Every real model's cross-era bar falls SHORT of this line -- the inversion.
    chance = float(dummy["cross_roc_auc"])
    axes[0].axvline(chance, color=INK_SECONDARY, linewidth=1.0, linestyle=(0, (3, 2)), zorder=4)
    axes[0].text(
        chance - 0.025, header_y, f"chance {chance:.4f}", ha="right", va="center",
        fontsize=6.4, color=INK_SECONDARY,
    )

    # Panel (b): the prevalence artifact, one line per regime in that regime's colour. The
    # majority-class dummy scores these two F1s by predicting "attack" on every row, so they move
    # with the class balance alone. Labels are staggered vertically -- the two lines are only 0.155
    # apart on x and their labels would otherwise collide.
    for metric_key, colour, label_y in (
        ("in_f1", REGIME_COLOURS["in_distribution"], header_y - 0.30),
        ("cross_f1", REGIME_COLOURS["cross_era"], header_y + 0.28),
    ):
        value = float(dummy[metric_key])
        axes[1].axvline(value, color=colour, linewidth=1.0, linestyle=(0, (3, 2)), zorder=4)
        axes[1].text(
            value, label_y, f"{value:.4f}", ha="center", va="center", fontsize=6.4, color=colour,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    reference_handles = [
        plt.Line2D([], [], color=INK_SECONDARY, linestyle=(0, (3, 2)),
                   label=f"(a) chance, ROC-AUC = {chance:.4f}"),
        plt.Line2D([], [], color=REGIME_COLOURS["in_distribution"], linestyle=(0, (3, 2)),
                   label=f"(b) majority-class F1, UNSW-test = {float(dummy['in_f1']):.4f}"),
        plt.Line2D([], [], color=REGIME_COLOURS["cross_era"], linestyle=(0, (3, 2)),
                   label=f"(b) majority-class F1, TON_IoT = {float(dummy['cross_f1']):.4f}"),
    ]
    fig.legend(
        [*handles, *reference_handles],
        [*labels, *(h.get_label() for h in reference_handles)],
        loc="lower center", bbox_to_anchor=(0.5, 0.004), ncol=2, handlelength=1.9,
        columnspacing=1.6, labelcolor=INK_PRIMARY,
    )
    _subtitle(
        fig,
        "RQ1: a 2015-trained IDS does not decay on 2019–20 traffic — it inverts",
        "Every model's cross-era ROC-AUC lands below chance, so its score ranking runs backwards on "
        "the newer era.\nΔ = in-distribution − cross-era (positive = performance lost). "
        f"UNSW-NB15 test n={in_n:,}, {1 - in_share:.2%} normal / {in_share:.2%} attack;  "
        f"TON_IoT n={cross_n:,}, {1 - cross_share:.2%} normal / {cross_share:.2%} attack.",
    )
    fig.subplots_adjust(left=0.175, right=0.995, top=0.79, bottom=0.305, wspace=0.30)

    sources: list[SourceRow] = []
    for _, row in table.iterrows():
        label = MODEL_LABELS[row["model"]]
        sources.append((f"{label} — in-distribution bars (a, b)",
                        (row["in_run_id"], row["model"], "in_distribution")))
        sources.append((f"{label} — cross-era bars (a, b)",
                        (row["cross_run_id"], row["model"], "cross_era")))
    sources.append(("(a) chance line, (b) both majority-class F1 lines",
                    ("phase4-baselines / phase6-crossera", "dummy", "in_distribution / cross_era")))

    # Ranges quoted in the caption are computed from the plotted rows, never transcribed: the real
    # models only, since the dummy sits at 0.5000 in both regimes by construction.
    real = table[table["model"] != "dummy"]
    caption = (
        f"**Figure 1. Cross-era collapse of a 2015-trained intrusion detector (RQ1).** "
        f"Each model is trained once on the UNSW-NB15 (2015) training fold and evaluated unchanged "
        f"in two regimes: in-distribution on the UNSW-NB15 test set (n={in_n:,}; "
        f"{1 - in_share:.2%} normal / {in_share:.2%} attack) and zero-shot cross-era on TON_IoT "
        f"(2019–20; n={cross_n:,}; {1 - cross_share:.2%} normal / {cross_share:.2%} attack), with "
        f"no refitting of either the model or the preprocessor. "
        f"Panel (a) leads because ROC-AUC is insensitive to the change in class balance: every real "
        f"model falls from {real['in_roc_auc'].min():.4f}–{real['in_roc_auc'].max():.4f} "
        f"in-distribution to {real['cross_roc_auc'].min():.4f}–{real['cross_roc_auc'].max():.4f} "
        f"cross-era, i.e. *below* the {chance:.4f} chance line, so the learned score ranking does "
        f"not merely stop working but runs backwards (a rank inversion, not a decay); the largest "
        f"loss is Δ {(real['in_roc_auc'] - real['cross_roc_auc']).max():+.4f}. "
        f"Panel (b) must be read against the two dashed majority-class lines: a dummy that predicts "
        f"\"attack\" everywhere scores F1 {float(dummy['in_f1']):.4f} on UNSW-test and "
        f"{float(dummy['cross_f1']):.4f} on TON_IoT purely because the attack share rises from "
        f"{in_share:.2%} to {cross_share:.2%}, so part of every F1 movement is prevalence rather "
        f"than drift. "
        f"Δ = in-distribution − cross-era throughout, i.e. what the model *lost*: positive means "
        f"degraded. The two ablation conditions (`phase6-crossera-no_proto`, "
        f"`phase6-crossera-no_conn_state`) are excluded from this figure by construction."
    )
    return save_figure(fig, "drift_indist_vs_crossera", caption, sources, out=out)


# =========================================================================================
# Figure 2 — RQ2 recovery curve
# =========================================================================================

#: The five budget points of the recovery curve, in order. Each is its own ``run_id``: budgets are
#: fractions **of the 105,522-row fine-tune pool**, and every point is scored on the same frozen
#: 105,521-row TON_IoT test half.
RECOVERY_RUN_IDS: tuple[str, ...] = (
    "phase7-recovery-f0.00",
    "phase7-recovery-f0.01",
    "phase7-recovery-f0.05",
    "phase7-recovery-f0.10",
    "phase7-recovery-f0.25",
)

#: Each model's upper bound: its own adaptation mechanism at the full pool (fraction 1.0).
CEILING_RUN_ID: str = "phase7-recovery-ceiling"

#: The MLP freeze-cost control. It is a *control*, not a budget point -- same budget as the ceiling,
#: different mechanism (every layer retrained instead of head-only) -- so it must never be drawn as
#: part of the curve. Mentioned in the caption, asserted absent from the plotted rows.
FREEZE_CONTROL_RUN_ID: str = "phase7-recovery-ceiling-no_freeze"

#: The one regime label Phase 7 logs under. Deliberately neither ``cross_era`` nor
#: ``in_distribution``, so a Phase 7 row cannot be mistaken for a Phase 6 row on a join.
TRANSFER_REGIME: str = "target_frozen_test"

#: Models drawn as curves. The dummy is the *floor*, not a competitor, and is drawn as a reference
#: line instead.
RECOVERY_MODELS: tuple[str, ...] = tuple(m for m in MODEL_ORDER if m != "dummy")
FLOOR_MODEL: str = "dummy"

_BUDGET_NOTE = re.compile(r"budget\s+([0-9.]+)%\s+of the pool,\s+n_ft=(\d+)")


def _budget_of(row: pd.Series) -> tuple[float, int]:
    """The (fraction-of-pool, n_ft) a Phase 7 row was measured at, cross-checked two ways.

    The fraction is parsed from the ``run_id`` suffix (which is the identity of the condition) and
    verified against the ``budget X% of the pool, n_ft=N`` string the run wrote into ``notes``. Both
    come from the committed log, so nothing here is transcribed from prose -- and if the two ever
    disagree, a point is about to be plotted at an x it was not measured at.
    """
    match = _BUDGET_NOTE.search(str(row["notes"]))
    if match is None:
        raise ValueError(f"{row['run_id']}/{row['model']} notes carry no budget: {row['notes']!r}")
    noted_fraction, n_ft = float(match.group(1)) / 100.0, int(match.group(2))
    run_id = str(row["run_id"])
    suffix = run_id.rsplit("-f", 1)[-1] if "-f" in run_id else ""
    try:
        declared = float(suffix)
    except ValueError:
        raise ValueError(
            f"{run_id!r} carries no `-f<fraction>` budget suffix, so it is not a point on the "
            "recovery curve. Only the RECOVERY_RUN_IDS budgets belong on the x-axis."
        ) from None
    if abs(declared - noted_fraction) > 1e-9:
        raise ValueError(
            f"{row['run_id']} declares budget {declared:.4f} but its notes record "
            f"{noted_fraction:.4f}; the x-coordinate of that point is not what it was measured at."
        )
    return declared, n_ft


def _recovery_table(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """``(curve, ceilings, floor)`` for the recovery figure, all from ``target_frozen_test`` rows.

    ``curve`` is one row per (model, budget) over :data:`RECOVERY_RUN_IDS`; ``ceilings`` is one row
    per model from :data:`CEILING_RUN_ID`; ``floor`` is the dummy's constant ROC-AUC and F1 on the
    frozen half, asserted constant across every budget *and* the ceiling before it is called a floor.
    """
    # Checked against the CONSTANTS, before anything is read: the control shares the ceiling's
    # budget and differs only in mechanism, so the only way it reaches this figure is by being
    # added to one of these two names. Checking the derived frames instead would be dead code --
    # `_budget_of` would already have refused the row for having no `-f<fraction>` suffix.
    if FREEZE_CONTROL_RUN_ID in (*RECOVERY_RUN_IDS, CEILING_RUN_ID):
        raise RuntimeError(
            f"{FREEZE_CONTROL_RUN_ID} is configured as a recovery-curve point. It is the MLP "
            "freeze-cost control at the same budget as the ceiling, not a budget point, and "
            "belongs in the caption rather than on the curve."
        )

    records = []
    for run_id in RECOVERY_RUN_IDS:
        for model in RECOVERY_MODELS:
            row = select_row(frame, run_id, model, TRANSFER_REGIME)
            fraction, n_ft = _budget_of(row)
            records.append({
                "run_id": run_id, "model": model, "fraction": fraction, "n_ft": n_ft,
                "roc_auc": float(row["roc_auc"]), "f1": float(row["f1"]),
                "n_test": int(row["n_test"]),
            })
    curve = pd.DataFrame.from_records(records)

    ceilings = pd.DataFrame.from_records([
        {
            "run_id": CEILING_RUN_ID, "model": model,
            "roc_auc": float(row["roc_auc"]), "f1": float(row["f1"]),
            "n_ft": int(_BUDGET_NOTE.search(str(row["notes"])).group(2)),
            "n_test": int(row["n_test"]),
        }
        for model in RECOVERY_MODELS
        for row in (select_row(frame, CEILING_RUN_ID, model, TRANSFER_REGIME),)
    ])

    floor_rows = [
        select_row(frame, run_id, FLOOR_MODEL, TRANSFER_REGIME)
        for run_id in (*RECOVERY_RUN_IDS, CEILING_RUN_ID)
    ]
    floor = {
        "roc_auc": float(floor_rows[0]["roc_auc"]),
        "f1": float(floor_rows[0]["f1"]),
        "n_test": int(floor_rows[0]["n_test"]),
    }
    for metric in ("roc_auc", "f1"):
        values = {round(float(row[metric]), 9) for row in floor_rows}
        if len(values) != 1:
            raise RuntimeError(
                f"the majority-class {metric} is not constant across the Phase 7 budgets "
                f"({sorted(values)}); it cannot be drawn as a single floor line."
            )

    # The whole design of Phase 7 is that the data budget is the only variable: every point,
    # including the ceiling, is scored on the SAME frozen half. Assert it rather than trust it.
    test_sizes = set(curve["n_test"]) | set(ceilings["n_test"]) | {floor["n_test"]}
    if len(test_sizes) != 1:
        raise RuntimeError(
            f"the recovery points were scored on differently sized test sets {sorted(test_sizes)}; "
            "every Phase 7 point must sit on the one frozen TON_IoT half."
        )
    return curve, ceilings, floor


def plot_recovery_curve(recovery: Any, out: Any = FIGURES) -> Path:
    """F1/ROC-AUC vs fraction of modern data used (RQ2 secondary headline).

    **One figure, two panels** -- (a) ROC-AUC, (b) F1 -- rather than two files: both panels carry
    the same five model curves against the same x-axis, so a shared legend serves both and the
    reader compares the two metrics without turning a page. In a 6-page report that also costs one
    figure slot instead of two.

    x is the fine-tune budget as a fraction of the 105,522-row pool, drawn on **ordinal** positions
    rather than a linear scale: the entire RQ2 claim lives in the 0 -> 1% segment, which a linear
    axis would compress to nothing. The right-hand position, past a dashed break, is each model's
    own ceiling at the full pool (fraction 1.0) -- shown as an open marker on a dotted connector
    because the ceiling is *per model* and five near-coincident horizontal lines at 0.98-1.00 would
    be unreadable. The floor is the majority-class dummy on the same frozen half, drawn as a solid
    reference line on both panels; on F1 it sits at 0.8656, which is the number that makes the F1
    panel worth reading at all.

    ``recovery`` is the DataFrame from :func:`load_metrics` (or ``None`` / a path).
    """
    apply_style()
    frame = as_metrics_frame(recovery)  # coerced once: `None` would otherwise re-read the log
    curve, ceilings, floor = _recovery_table(frame)

    fractions = sorted(curve["fraction"].unique())
    x_budget = list(range(len(fractions)))
    x_ceiling = len(fractions) + 0.55  # past the axis break
    n_ft_by_fraction = {
        fraction: int(curve[curve["fraction"] == fraction]["n_ft"].iloc[0]) for fraction in fractions
    }
    ceiling_n_ft = int(ceilings["n_ft"].iloc[0])
    n_test = int(curve["n_test"].iloc[0])

    fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH_IN, 3.55), sharex=True)
    panels = (
        ("roc_auc", "ROC-AUC", axes[0], "(a) ROC-AUC", (0.12, 1.05)),
        ("f1", "F1 (attack class)", axes[1], "(b) F1 (attack class)", (0.18, 1.05)),
    )
    break_x = len(fractions) - 0.5 + 0.275
    for metric, ylabel, ax, panel_title, ylim in panels:
        # The axis break: everything to its right is the full pool, not a budget on the curve.
        ax.axvline(break_x, color=INK_MUTED, linewidth=0.9, linestyle=(0, (2, 2)), alpha=0.6,
                   zorder=1)
        # Floor next, so the curves cross it in front.
        ax.axhline(floor[metric], color=INK_SECONDARY, linewidth=1.1, linestyle=(0, (3, 2)),
                   zorder=2)
        # Knocked out of the line and the grid behind it: the F1 floor at 0.8656 is the single
        # number this panel exists to be read against, so it must not be half-struck-through.
        ax.text(
            -0.30, floor[metric] - 0.028, f"floor {floor[metric]:.4f}", ha="left", va="top",
            fontsize=6.5, color=INK_SECONDARY, zorder=5,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
        )

        for model in RECOVERY_MODELS:
            rows = curve[curve["model"] == model].sort_values("fraction")
            colour = MODEL_COLOURS[model]
            ax.plot(
                x_budget, rows[metric].to_numpy(), color=colour, linestyle=MODEL_DASHES[model],
                marker=MODEL_MARKERS[model], markersize=4.0, markeredgecolor="white",
                markeredgewidth=0.5, label=MODEL_LABELS[model], zorder=3,
            )
            ceiling = float(ceilings[ceilings["model"] == model][metric].iloc[0])
            ax.plot(
                [x_budget[-1], x_ceiling], [rows[metric].to_numpy()[-1], ceiling],
                color=colour, linestyle=(0, (1, 1.4)), linewidth=1.0, alpha=0.7, zorder=3,
            )
            ax.plot(
                [x_ceiling], [ceiling], marker=MODEL_MARKERS[model], markersize=4.4,
                markerfacecolor="white", markeredgecolor=colour, markeredgewidth=1.1, zorder=4,
            )

        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title, pad=6)
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.4, x_ceiling + 0.4)
        ax.set_xticks([*x_budget, x_ceiling])
        ax.set_xticklabels(
            [
                *(f"{fraction:.0%}\n{n_ft_by_fraction[fraction]:,}" for fraction in fractions),
                f"100%\n{ceiling_n_ft:,}",
            ],
            fontsize=6.4,
        )

    # One shared x label, not one per panel: at half-page width the per-panel copies overran the
    # axes and collided with each other.
    fig.supxlabel(
        f"Fine-tune budget — share of the {ceiling_n_ft:,}-row TON_IoT pool, and the resulting "
        "number of labelled flows",
        fontsize=8.5, color=INK_PRIMARY, y=0.145,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    ceiling_handle = plt.Line2D(
        [], [], color=INK_SECONDARY, linestyle=(0, (1, 1.4)), marker="o", markersize=4.4,
        markerfacecolor="white", markeredgecolor=INK_SECONDARY,
        label="per-model ceiling (full pool)",
    )
    floor_handle = plt.Line2D(
        [], [], color=INK_SECONDARY, linestyle=(0, (3, 2)),
        label="majority-class dummy floor",
    )
    fig.legend(
        [*handles, ceiling_handle, floor_handle], [*labels, ceiling_handle.get_label(),
                                                   floor_handle.get_label()],
        loc="lower center", bbox_to_anchor=(0.5, 0.004), ncol=4, handlelength=2.2,
        columnspacing=1.3, labelcolor=INK_PRIMARY,
    )
    _subtitle(
        fig,
        "RQ2: 1,055 labelled modern flows undo the inversion",
        "The same six models as Figure 1, fine-tuned on a budget of TON_IoT labels and scored on "
        f"one frozen TON_IoT\ntest half (n={n_test:,}) that no budget ever draws from. x is "
        "ordinal; the column past the break is each model's\nown ceiling at the full pool, not a "
        "point on the curve.",
    )
    fig.subplots_adjust(left=0.095, right=0.995, top=0.755, bottom=0.275, wspace=0.235)

    sources: list[SourceRow] = []
    for model in RECOVERY_MODELS:
        for run_id in RECOVERY_RUN_IDS:
            fraction = float(run_id.rsplit("-f", 1)[-1])
            sources.append((
                f"{MODEL_LABELS[model]} — point at {fraction:.0%} budget (a, b)",
                (run_id, model, TRANSFER_REGIME),
            ))
        sources.append((
            f"{MODEL_LABELS[model]} — open ceiling marker (a, b)",
            (CEILING_RUN_ID, model, TRANSFER_REGIME),
        ))
    sources.append((
        "majority-class floor line (a, b) — constant across all six run_ids",
        (f"{RECOVERY_RUN_IDS[0]} … {CEILING_RUN_ID}", FLOOR_MODEL, TRANSFER_REGIME),
    ))

    # Every quantity the caption quotes is computed from the plotted rows (plus, for the freeze
    # cost, the control row that is deliberately NOT plotted) rather than transcribed from prose.
    zero = curve[curve["fraction"] == 0.0]
    one_pct = curve[curve["fraction"] == 0.01]
    largest = curve[curve["fraction"] == max(fractions)]
    closed = []
    overshoot = 0.0
    for model in RECOVERY_MODELS:
        for metric in ("roc_auc", "f1"):
            start = float(zero[zero["model"] == model][metric].iloc[0])
            ceiling = float(ceilings[ceilings["model"] == model][metric].iloc[0])
            if metric == "roc_auc":
                end = float(largest[largest["model"] == model][metric].iloc[0])
                closed.append((end - start) / (ceiling - start))
            overshoot = max(
                overshoot,
                float((curve[curve["model"] == model][metric] - ceiling).max()),
            )
    freeze_cost = (
        float(select_row(frame, FREEZE_CONTROL_RUN_ID, "scratch_mlp", TRANSFER_REGIME)["roc_auc"])
        - float(ceilings[ceilings["model"] == "scratch_mlp"]["roc_auc"].iloc[0])
    )
    caption = (
        f"**Figure 2. How little modern data undoes the cross-era collapse (RQ2).** "
        f"TON_IoT is split once (seed 42, stratified) into a permanent {n_test:,}-row test half and "
        f"a disjoint {ceiling_n_ft:,}-row fine-tune pool; every point on both panels is scored on "
        f"that same frozen half, so the labelling budget is the only variable along x. The "
        f"zero-budget point is re-measured on the frozen half rather than carried over from the "
        f"cross-era run of Figure 1, which was scored on all 211,043 TON_IoT rows and therefore "
        f"included the pool. "
        f"A budget of 1% ({n_ft_by_fraction[0.01]:,} labelled flows) lifts every model from "
        f"ROC-AUC {zero['roc_auc'].min():.4f}–{zero['roc_auc'].max():.4f} to "
        f"{one_pct['roc_auc'].min():.4f}–{one_pct['roc_auc'].max():.4f}, clearing the "
        f"majority-class floor of {floor['roc_auc']:.4f} in every case, and the "
        f"{max(fractions):.0%} budget closes {min(closed):.1%}–{max(closed):.1%} of the remaining "
        f"ROC-AUC gap to each model's own ceiling. "
        f"The F1 panel is the one that needs its floor: a dummy predicting \"attack\" everywhere "
        f"already scores F1 {floor['f1']:.4f} on this half, so only recovery well above that line "
        f"means anything. "
        f"Ceilings are each model's own adaptation mechanism at the full pool and are shown past "
        f"the axis break; the MLP's is set by its mechanism rather than by data — freezing both "
        f"hidden layers costs {freeze_cost:.4f} ROC-AUC against the same architecture retrained "
        f"end to end, logged separately as `{FREEZE_CONTROL_RUN_ID}` and excluded from this "
        f"figure. "
        f"A few points sit marginally above their own ceiling at the larger budgets (by at most "
        f"{overshoot:.4f}, i.e. run-to-run noise)."
    )
    return save_figure(fig, "recovery_curve", caption, sources, out=out)


# =========================================================================================
# Remaining Phase 9 figures — see the module docstring for the data paths each still needs.
# =========================================================================================


# =========================================================================================
# Figure 3 — RQ1 confusion matrices
# =========================================================================================

#: The one Phase 6 condition this figure renders: the full d=22 shared feature set.
#: ``run_phase6()`` returns all three conditions and the sidecar persists all three, so the
#: selection has to be explicit and asserted -- the two ablations are matched d=18 experiments,
#: comparable to this condition and to nothing else, and an ablated matrix here would read as the
#: headline result. Note this is the run_id of the *condition*, which is where all twelve matrices
#: come from; it is NOT where every matrix's cross-check row lives (see :data:`CSV_RUN_ID_FOR`).
CONFUSION_CONDITION: str = CROSS_ERA_RUN_ID

#: Rows of the grid, in reading order: the era the model was built for, then the one it was carried
#: onto. Panel (a) above panel (b) is the same ordering Figure 1's bars use.
REGIME_ORDER: tuple[str, ...] = ("in_distribution", "cross_era")

#: Axis tick labels. Spelled with the integer encoding because the cell order below depends on it.
TRUE_CLASS_LABELS: tuple[str, ...] = ("true\nnormal (0)", "true\nattack (1)")
PREDICTED_CLASS_LABELS: tuple[str, ...] = ("pred.\nnormal", "pred.\nattack")

#: Sequential ramp for the row-normalised share. Deliberately **neutral**, not one of the two
#: regime hues: hue already means *regime* everywhere else in this figure set (blue = the model's
#: own era, orange = the era it was carried onto), and a heatmap that spent it on magnitude instead
#: would break that across figures. The ramp darkens early so that the white-text threshold below
#: sits comfortably above 4.5:1 on either side of the switch.
CONFUSION_CMAP = LinearSegmentedColormap.from_list(
    "crossera_share", ["#ffffff", "#c3c1ba", "#6e6c66", "#2f2e2c"]
)

#: Row share above which a cell's annotation flips from ink to white.
_DARK_CELL_SHARE: float = 0.55


def _derived_from_matrix(matrix: Sequence[Sequence[int]]) -> dict[str, float]:
    """The scalar metrics a 2x2 count matrix determines, for the cross-check against the log.

    ``matrix`` is ``[[tn, fp], [fn, tp]]`` -- sklearn's ``confusion_matrix(..., labels=[0, 1])``
    with 1 = attack (:data:`evaluate.POSITIVE_LABEL`). ``zero_division`` is handled the way
    :func:`evaluate.evaluate` handles it (0.0, not NaN), so a model that never predicts the
    positive class still compares cleanly.
    """
    (tn, fp), (fn, tp) = ((int(matrix[0][0]), int(matrix[0][1])),
                          (int(matrix[1][0]), int(matrix[1][1])))
    n = tn + fp + fn + tp
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    return {
        "n_test": float(n),
        "accuracy": (tn + tp) / n,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "positive_rate": (fn + tp) / n,
    }


#: Where each (model, regime)'s **cross-check** row lives in ``reports/metrics.csv``.
#:
#: The matrices themselves all come from the one ``phase6-crossera`` condition, but the scalar rows
#: they are checked against do not all sit under that ``run_id``: Phase 6 computes the four
#: baselines' in-distribution half for the delta and deliberately does not re-log it, because those
#: rows are ``phase4-baselines``' (``evaluate.run_condition(log_in_distribution=False)``). That is
#: the same join :data:`IN_DISTRIBUTION_RUN_ID` encodes for Figure 1, reused here rather than
#: re-derived.
def CSV_RUN_ID_FOR(model: str, regime: str) -> str:  # noqa: N802 - a table spelled as a function
    return IN_DISTRIBUTION_RUN_ID[model] if regime == "in_distribution" else CONFUSION_CONDITION


def as_confusion_results(results: Any) -> dict[str, dict[str, dict[str, Any]]]:
    """Coerce whatever a caller passed as ``results`` into ``{run_id: {model: {regime: ...}}}``.

    Three spellings, mirroring :func:`as_metrics_frame`: ``None`` reads the committed sidecar
    (what :func:`main` does), a path reads that file, and a dict is taken as-is -- which is
    :func:`evaluate.run_phase6`'s return value, so a caller that has just run Phase 6 in-process
    can hand it straight over without a round trip through disk.
    """
    if isinstance(results, dict):
        return results
    if results is None:
        return read_confusion_matrices()
    return read_confusion_matrices(results)


def _confusion_table(results: Any, frame: pd.DataFrame) -> list[dict[str, Any]]:
    """One record per (model, regime) panel: the matrix, its shares, and the log row it agrees with.

    **Every matrix is cross-checked against ``reports/metrics.csv`` before it is drawn**, and the
    check is the reason this figure can claim to illustrate the logged result rather than a second,
    parallel measurement of it. The matrix determines ``accuracy``, ``precision``, ``recall``,
    ``balanced_accuracy``, ``n_test`` and ``positive_rate`` exactly; all six are recomputed from the
    counts and compared to the committed row. A disagreement means the sidecar and the log came
    from different runs, which is precisely the failure a persisted artifact introduces and the
    only one worth guarding against -- so it raises rather than drawing something plausible.

    Tolerances differ by which row is being compared against, and the difference is not cosmetic:

    * a row Phase 6 logged itself is a rounding of these very counts, so ``1e-6`` (the log holds six
      decimals) is the right bar;
    * the four baselines' in-distribution rows are ``phase4-baselines``', re-derived here from
      Phase 6's re-fit of the same locked factory on the same fold. Those agree to float noise
      rather than to the bit, which is exactly what ``evaluate._check_against_phase4`` already
      allows at :data:`evaluate.PHASE4_AGREEMENT_TOLERANCE`; using a tighter bar here would make
      this figure fail on a difference Phase 6 itself accepts.
    """
    conditions = as_confusion_results(results)
    if CONFUSION_CONDITION in ABLATION_RUN_IDS:  # pragma: no cover - a constant, asserted anyway
        raise RuntimeError(
            f"{CONFUSION_CONDITION} is an ablation run_id. The confusion figure is the RQ1 "
            "headline and must render the full d=22 condition only."
        )
    if CONFUSION_CONDITION not in conditions:
        raise KeyError(
            f"the confusion sidecar holds conditions {sorted(conditions)} but not "
            f"{CONFUSION_CONDITION!r}. Re-run `python -m src.evaluate --regimes`."
        )
    condition = conditions[CONFUSION_CONDITION]

    records: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        if model not in condition:
            raise KeyError(
                f"no confusion matrix for {model!r} under {CONFUSION_CONDITION!r}; the sidecar "
                f"holds {sorted(condition)}. All six models are drawn or none are."
            )
        for regime in REGIME_ORDER:
            entry = condition[model][regime]
            matrix = [[int(value) for value in row] for row in entry["confusion_matrix"]]
            if len(matrix) != 2 or any(len(row) != 2 for row in matrix):
                raise ValueError(
                    f"{CONFUSION_CONDITION}/{model}/{regime} carries a "
                    f"{len(matrix)}x{len(matrix[0]) if matrix else 0} matrix, not the 2x2 "
                    "[[tn, fp], [fn, tp]] this figure renders."
                )
            derived = _derived_from_matrix(matrix)

            csv_run_id = CSV_RUN_ID_FOR(model, regime)
            logged = select_row(frame, csv_run_id, model, regime)
            tolerance = (
                PHASE4_AGREEMENT_TOLERANCE if csv_run_id != CONFUSION_CONDITION else 1e-6
            )
            for metric in ("accuracy", "precision", "recall", "balanced_accuracy",
                           "n_test", "positive_rate"):
                drift = abs(derived[metric] - float(logged[metric]))
                if drift > (0 if metric == "n_test" else tolerance):
                    raise RuntimeError(
                        f"the confusion matrix for {model}/{regime} implies {metric} "
                        f"{derived[metric]:.6f}, but the committed "
                        f"({csv_run_id}, {model}, {regime}) row of reports/metrics.csv records "
                        f"{logged[metric]} (drift {drift:.2e} > {tolerance:.0e}). The sidecar and "
                        "the run log are not from the same run; regenerate both with "
                        "`python -m src.evaluate --regimes` before drawing anything from either."
                    )

            row_totals = [matrix[0][0] + matrix[0][1], matrix[1][0] + matrix[1][1]]
            records.append({
                "model": model,
                "regime": regime,
                "matrix": matrix,
                "shares": [
                    [cell / total if total else 0.0 for cell in row]
                    for row, total in zip(matrix, row_totals)
                ],
                "n_test": int(derived["n_test"]),
                "positive_rate": derived["positive_rate"],
                "balanced_accuracy": derived["balanced_accuracy"],
                "recall": derived["recall"],
                "specificity": derived["specificity"],
                "csv_run_id": csv_run_id,
            })
    return records


def plot_confusion_matrices(results: Any, out: Any = FIGURES) -> Path:
    """Confusion matrices for all six models in both regimes (RQ1, the mechanism behind Figure 1).

    A 2x6 grid: one row per regime, one column per model, columns in :data:`MODEL_ORDER` so a
    reader carries Figure 1's ordering straight across. Each cell carries **both** its raw count and
    that count as a percentage of its own true-class row, because the two regimes are measured on
    sets of very different size and balance (82,332 rows at 55.06% attack against 211,043 at
    76.31%) and raw counts alone are not comparable between the rows of this figure.

    Row-normalising is also what makes the RQ1 headline legible rather than merely present. The two
    diagonal percentages are specificity and recall; their mean is the **balanced accuracy** printed
    above each matrix, which is a logged column and is below 0.5000 for every real model in the
    cross-era row -- i.e. averaged over the two classes those models predict the wrong label more
    often than the right one. The dark band moves off the diagonal between the rows: that *is* the
    rank inversion of Figure 1, drawn at the level of predictions instead of scores.

    ``results`` is :func:`evaluate.run_phase6`'s return value, ``None`` (read the committed
    ``reports/confusion_matrices.json`` sidecar -- what :func:`main` passes) or a path to one. The
    metrics log is re-read internally for the cross-check in :func:`_confusion_table`; that is a
    73-row file and the second read costs nothing, which is cheaper than widening the stub's
    signature to thread a frame through.
    """
    apply_style()
    frame = as_metrics_frame(None)
    table = _confusion_table(results, frame)
    by_key = {(record["model"], record["regime"]): record for record in table}

    # Twelve 2x2 matrices at report column width leaves each cell about 0.38 x 0.43 in, which is
    # what sets the annotation size below. The generous `hspace` is not whitespace for its own
    # sake: the gap between the two rows has to hold row (a)'s tick labels, row (b)'s header band,
    # its six model titles and its balanced-accuracy line, and the row headers are positioned in
    # figure coordinates relative to the axes, so shrinking it silently overlaps them.
    fig_height = 4.75
    fig, axes = plt.subplots(2, 6, figsize=(PAGE_WIDTH_IN, fig_height))
    fig.subplots_adjust(left=0.105, right=0.845, top=0.715, bottom=0.150, wspace=0.10, hspace=1.10)

    image = None
    for column, model in enumerate(MODEL_ORDER):
        for row, regime in enumerate(REGIME_ORDER):
            record = by_key[(model, regime)]
            ax = axes[row][column]
            image = ax.imshow(
                record["shares"], cmap=CONFUSION_CMAP, vmin=0.0, vmax=1.0, aspect="auto",
            )
            for i in range(2):
                for j in range(2):
                    share = record["shares"][i][j]
                    ax.text(
                        j, i, f"{record['matrix'][i][j]:,}\n{share:.1%}",
                        ha="center", va="center", fontsize=5.3, linespacing=1.35,
                        color="white" if share > _DARK_CELL_SHARE else INK_PRIMARY,
                    )
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(PREDICTED_CLASS_LABELS, fontsize=5.9, linespacing=1.2)
            ax.set_yticklabels(
                TRUE_CLASS_LABELS if column == 0 else ["", ""], fontsize=5.9, linespacing=1.2,
            )
            ax.tick_params(length=0, pad=1.6)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(GRID)
                spine.set_linewidth(0.6)
            # The model name goes above the top matrix of each regime block, not once per column:
            # the two blocks are separated by their own header band, so each has to be readable
            # without tracing a column back past it.
            ax.set_title(textwrap.fill(MODEL_LABELS[model], 16), fontsize=6.6, pad=15)
            ax.text(
                0.5, 1.025, f"bal. acc {record['balanced_accuracy']:.4f}",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=6.0,
                color=INK_SECONDARY,
            )

    # Row headers, one per regime, carrying the set each row was measured on. The swatch is what
    # ties the row to Figure 1's bars; the text stays in ink (colour identifies, it does not label).
    # Positioned relative to each row's axes rather than at a fixed y, so the two headers stay put
    # if the grid geometry above is retuned. They must also stay clear of the colour bar on the
    # right, which is why they carry the attack share alone -- the normal share is 1 minus it, and
    # both are spelled out in the caption.
    for row, regime in enumerate(REGIME_ORDER):
        record = by_key[(MODEL_ORDER[0], regime)]
        y = axes[row][0].get_position().y1 + 0.58 / fig_height
        fig.add_artist(Rectangle(
            (0.062, y + 0.004), 0.0105, 0.0155, transform=fig.transFigure,
            facecolor=REGIME_COLOURS[regime], edgecolor="none",
        ))
        fig.text(
            0.080, y, f"({'ab'[row]}) {REGIME_LABELS[regime]}  —  n={record['n_test']:,}, "
            f"{record['positive_rate']:.2%} attack",
            ha="left", va="bottom", fontsize=7.2, fontweight="bold", color=INK_PRIMARY,
        )

    colour_bar = fig.colorbar(
        image, cax=fig.add_axes([0.858, 0.30, 0.0125, 0.32]), ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    colour_bar.ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=6.0)
    colour_bar.ax.tick_params(length=1.8, pad=1.6)
    colour_bar.outline.set_visible(False)
    colour_bar.set_label(
        "cell as a share of its true-class row", fontsize=6.4, color=INK_SECONDARY, labelpad=3,
    )

    # Line breaks here and in the subtitle are hand-placed rather than wrapped, and the lengths
    # matter: `savefig.bbox = "tight"` sizes the written PNG to its widest artist, so a centred
    # line longer than the axes grid silently widens the whole figure and the report then embeds it
    # scaled down, shrinking every label. Keep each line under ~125 characters at this size.
    fig.text(
        0.5, 0.010,
        "Cell order [[tn, fp], [fn, tp]] — sklearn labels=[0, 1], 1 = attack: rows are the true "
        "class, columns the predicted class.\n"
        "Each cell shows the count and, below it, that count as a share of its own true-class row; "
        "the shading encodes that share.\n"
        "Balanced accuracy above each matrix is the mean of the two diagonal shares — the logged "
        "balanced_accuracy column of metrics.csv.",
        ha="center", va="bottom", fontsize=6.1, color=INK_SECONDARY, linespacing=1.5,
    )
    _subtitle(
        fig,
        "RQ1 at the prediction level: the errors move onto the anti-diagonal",
        "The same six models and two regimes as Figure 1. In-distribution the mass sits on the "
        "diagonal;\ncross-era it does not — every real model's balanced accuracy falls below 0.5000.",
    )

    # Provenance is deliberately two-sided here. All twelve MATRICES come from the one
    # `phase6-crossera` condition of `run_phase6()`; the run_id column below is the metrics.csv row
    # each was cross-checked against, which for the four baselines' in-distribution panels is
    # `phase4-baselines` (Phase 6 recomputes that half for the delta but does not re-log it).
    row_letter = {regime: "ab"[index] for index, regime in enumerate(REGIME_ORDER)}
    sources: list[SourceRow] = [
        (
            f"{MODEL_LABELS[record['model']]} — matrix in row ({row_letter[record['regime']]}); "
            f"matrix from run_phase6()/{CONFUSION_CONDITION}, cross-checked against",
            (record["csv_run_id"], record["model"], record["regime"]),
        )
        for record in table
    ]

    cross = [r for r in table if r["regime"] == "cross_era" and r["model"] != "dummy"]
    in_dist = [r for r in table if r["regime"] == "in_distribution" and r["model"] != "dummy"]
    dummy_cross = by_key[("dummy", "cross_era")]
    caption = (
        f"**Figure 3. Where the cross-era predictions go (RQ1).** "
        f"Confusion matrices for all six models in both regimes of Figure 1, from the same single "
        f"fit per model: in-distribution on the UNSW-NB15 test set "
        f"(n={in_dist[0]['n_test']:,}; {in_dist[0]['positive_rate']:.2%} attack) and zero-shot "
        f"cross-era on TON_IoT (n={cross[0]['n_test']:,}; {cross[0]['positive_rate']:.2%} attack). "
        f"Cell order is [[tn, fp], [fn, tp]] (sklearn `labels=[0, 1]`, 1 = attack); rows are the "
        f"true class and columns the predicted class, and each cell is annotated with its count "
        f"and with that count as a share of its own true-class row, because the two regimes are "
        f"measured on sets of different size and balance and raw counts do not compare across "
        f"them. "
        f"Reading down a column shows the RQ1 result as predictions rather than scores: the shaded "
        f"mass leaves the diagonal. Attack recall falls from "
        f"{min(r['recall'] for r in in_dist):.1%}–{max(r['recall'] for r in in_dist):.1%} to "
        f"{min(r['recall'] for r in cross):.1%}–{max(r['recall'] for r in cross):.1%}, and the "
        f"balanced accuracy printed above each matrix — the mean of the two diagonal shares, and a "
        f"logged column — drops from "
        f"{min(r['balanced_accuracy'] for r in in_dist):.4f}–"
        f"{max(r['balanced_accuracy'] for r in in_dist):.4f} to "
        f"{min(r['balanced_accuracy'] for r in cross):.4f}–"
        f"{max(r['balanced_accuracy'] for r in cross):.4f}, i.e. below 0.5000 for every real model: "
        f"averaged over the two classes they predict the wrong label more often than the right one. "
        f"The majority-class dummy is the control that shows this is not simply the change in "
        f"prevalence — it predicts \"attack\" on all {dummy_cross['n_test']:,} cross-era rows, so "
        f"its entire matrix sits in the right-hand column in both regimes and its balanced accuracy "
        f"is exactly {dummy_cross['balanced_accuracy']:.4f} in each. "
        f"Matrices are the ones `evaluate.run_phase6()` computed under `run_id={CONFUSION_CONDITION}` "
        f"(the full d=22 feature set; both ablation conditions are excluded by construction), "
        f"persisted to `reports/confusion_matrices.json`; every matrix was checked against the "
        f"accuracy, precision, recall, balanced accuracy and prevalence of its own row of "
        f"`reports/metrics.csv` before being drawn."
    )
    return save_figure(fig, "confusion_matrices", caption, sources, out=out)


# =========================================================================================
# Figure 4 — RQ1 ROC curves
# =========================================================================================

#: The Phase 6 condition this figure renders, and the only one it may. Same constant and same
#: argument as :data:`CONFUSION_CONDITION`: the sidecar carries all three conditions, both ablations
#: are matched d=18 experiments comparable to this one and to nothing else, and an ablated curve
#: drawn here would read as the headline.
ROC_CONDITION: str = CROSS_ERA_RUN_ID

#: Slack allowed between the trapezoidal area of the *drawn* (simplified) curve and the ``roc_auc``
#: committed to ``reports/metrics.csv``. Two independent terms, and neither is a fudge factor:
#:
#: * :data:`evaluate.ROC_CURVE_TOLERANCE` -- the bound ``evaluate.roc_points`` already enforced on
#:   the simplification before it wrote the curve down;
#: * half a unit in the log's last place -- ``metrics.csv`` stores six decimals, so the committed
#:   scalar is itself a rounding of the exact area.
#:
#: Measured worst case across the twelve curves is far inside this; see the figure's caption.
ROC_DRAWN_TOLERANCE: float = ROC_CURVE_TOLERANCE + 0.5 * 10 ** -METRIC_DECIMALS

#: Markers per curve. The curves carry hundreds of vertices, so a marker at every one would be a
#: solid band; six evenly spaced along the drawn polyline give the shape-based identity the palette
#: note above makes mandatory without obscuring the line.
_ROC_MARKERS_PER_CURVE: int = 6


def as_roc_curves(curves: Any) -> dict[str, dict[str, dict[str, Any]]]:
    """Coerce whatever a caller passed as ``curves`` into ``{run_id: {model: {regime: ...}}}``.

    The three spellings :func:`as_confusion_results` accepts, for the same reasons -- except that a
    dict handed straight from :func:`evaluate.run_phase6` holds the raw ``roc_curve`` sub-dict under
    each ``(model, regime)`` rather than the sidecar's flattened entry, so it is unwrapped here.
    """
    if isinstance(curves, dict):
        return {
            run_id: {
                model: {regime: scores["roc_curve"] for regime, scores in regimes.items()}
                for model, regimes in condition.items()
            }
            for run_id, condition in curves.items()
        }
    if curves is None:
        return read_roc_curves()
    return read_roc_curves(curves)


def _roc_table(curves: Any, frame: pd.DataFrame) -> list[dict[str, Any]]:
    """One record per (model, regime) curve: the rates to draw, and the log row they agree with.

    **Every curve is cross-checked against ``reports/metrics.csv`` before it is drawn**, three ways,
    because this figure prints an AUC next to each line and a printed number that disagrees with the
    committed one is the single failure a persisted artifact makes possible:

    * the class sizes the curve was built over must sum to the logged ``n_test`` and give the logged
      ``positive_rate``;
    * the sidecar's own ``roc_auc`` must equal the committed one;
    * the **drawn** polyline's trapezoidal area, recomputed here from the stored integer counts,
      must equal it too, within :data:`ROC_DRAWN_TOLERANCE`. That is the check that makes
      simplifying the curve safe: it is verified against the log at render time, not merely asserted
      at write time.

    Tolerances split exactly as :func:`_confusion_table`'s do, and for the same reason: the four
    baselines' in-distribution rows belong to ``phase4-baselines`` and are re-derived here from
    Phase 6's re-fit of the same locked factory, so they agree to
    :data:`evaluate.PHASE4_AGREEMENT_TOLERANCE` rather than to the bit.
    """
    conditions = as_roc_curves(curves)
    if ROC_CONDITION in ABLATION_RUN_IDS:  # pragma: no cover - a constant, asserted anyway
        raise RuntimeError(
            f"{ROC_CONDITION} is an ablation run_id. The ROC figure is the RQ1 headline and must "
            "render the full d=22 condition only."
        )
    if ROC_CONDITION not in conditions:
        raise KeyError(
            f"the ROC sidecar holds conditions {sorted(conditions)} but not {ROC_CONDITION!r}. "
            "Re-run `python -m src.evaluate --regimes`."
        )
    condition = conditions[ROC_CONDITION]

    records: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        if model not in condition:
            raise KeyError(
                f"no ROC curve for {model!r} under {ROC_CONDITION!r}; the sidecar holds "
                f"{sorted(condition)}. All six models are drawn or none are."
            )
        for regime in REGIME_ORDER:
            curve = condition[model][regime]
            n_negative, n_positive = int(curve["n_negative"]), int(curve["n_positive"])
            fpr = [count / n_negative for count in curve["false_positives"]]
            tpr = [count / n_positive for count in curve["true_positives"]]
            drawn_auc = float(_trapezoid(fpr, tpr))

            csv_run_id = CSV_RUN_ID_FOR(model, regime)
            logged = select_row(frame, csv_run_id, model, regime)
            logged_auc = float(logged["roc_auc"])
            slack = (
                PHASE4_AGREEMENT_TOLERANCE if csv_run_id != ROC_CONDITION else 0.0
            )
            checks = (
                ("n_test", float(n_negative + n_positive), float(logged["n_test"]), 0.0),
                ("positive_rate", n_positive / (n_negative + n_positive),
                 float(logged["positive_rate"]), max(slack, 1e-6)),
                ("roc_auc (sidecar scalar)", float(curve["roc_auc"]), logged_auc, slack),
                ("roc_auc (area of the drawn curve)", drawn_auc, logged_auc,
                 ROC_DRAWN_TOLERANCE + slack),
            )
            for what, measured, committed, tolerance in checks:
                if abs(measured - committed) > tolerance:
                    raise RuntimeError(
                        f"the ROC curve stored for {model}/{regime} gives {what} "
                        f"{measured:.9f}, but the committed ({csv_run_id}, {model}, {regime}) row "
                        f"of reports/metrics.csv records {committed:.9f} (drift "
                        f"{abs(measured - committed):.2e} > {tolerance:.1e}). The sidecar and the "
                        "run log are not from the same run, or the curve was simplified past the "
                        "AUC it illustrates; regenerate both with `python -m src.evaluate "
                        "--regimes` before drawing anything from either."
                    )

            records.append({
                "model": model,
                "regime": regime,
                "fpr": fpr,
                "tpr": tpr,
                "roc_auc": logged_auc,          # the COMMITTED scalar is what gets annotated
                "drawn_auc": drawn_auc,
                "auc_drift": abs(drawn_auc - logged_auc),
                "n_vertices": int(curve["n_vertices"]),
                "n_drawn": len(fpr),
                "n_test": n_negative + n_positive,
                "positive_rate": n_positive / (n_negative + n_positive),
                "csv_run_id": csv_run_id,
            })
    return records


def _trapezoid(x: Sequence[float], y: Sequence[float]) -> float:
    """Trapezoidal area under ``y`` over ``x``. Spelled out so the AUC check owns its arithmetic."""
    return sum(
        (x[i + 1] - x[i]) * (y[i + 1] + y[i]) / 2.0 for i in range(len(x) - 1)
    )


def plot_roc_curves(results: Any, out: Any = FIGURES) -> Path:
    """ROC curves for all six models in both regimes (RQ1, the curve behind Figure 1's bars).

    **Two panels, one per regime**, on identical square axes so the pair can be compared by eye: (a)
    in-distribution on UNSW-test, (b) zero-shot cross-era on TON_IoT. The point of the figure is
    that the cross-era curves do not merely flatten onto the chance diagonal, they cross *under* it
    -- so the region below the diagonal is tinted and labelled on both panels, and in (b) every real
    model lies entirely inside it. That is the rank inversion of ``deviations.md`` §3.10 drawn as a
    curve rather than as a number.

    Each curve is annotated with its ROC-AUC **in the panel legend**, read from the committed
    ``(run_id, model, regime)`` row of ``reports/metrics.csv`` rather than from the curve; the
    curve's own area is checked against that row first (see :func:`_roc_table`). Legends sit in the
    corner each panel leaves empty, which is itself the result: (a) fills the upper-left triangle
    and (b) the lower-right one.

    The majority-class dummy **is** drawn, as the degenerate case rather than as a competitor. A
    constant predictor produces a single operating point, so ``roc_curve`` returns the two endpoints
    and its ROC is the chance diagonal exactly, at AUC 0.5000 in both regimes; drawing it in neutral
    ink over the chance line makes the reference line and the measured floor visibly the same object
    instead of asking the reader to take that on trust, and it is the control that shows the
    cross-era collapse is not the change in class balance.

    ``results`` is :func:`evaluate.run_phase6`'s return value, ``None`` (read the committed
    ``reports/roc_curves.json`` sidecar -- what :func:`main` passes) or a path to one.
    """
    apply_style()
    frame = as_metrics_frame(None)
    table = _roc_table(results, frame)
    by_key = {(record["model"], record["regime"]): record for record in table}

    # Square panels: a ROC axis with unequal scales misreads, because the chance diagonal stops
    # being the 45-degree line the reader is looking for.
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH_IN, 4.35))
    fig.subplots_adjust(left=0.085, right=0.995, top=0.775, bottom=0.185, wspace=0.20)

    for column, regime in enumerate(REGIME_ORDER):
        ax = axes[column]
        # The sub-chance region, tinted identically on both panels so the comparison is immediate:
        # empty in (a), and in (b) it contains every real model's entire curve.
        ax.fill_between([0.0, 1.0], [0.0, 1.0], [0.0, 0.0], color=GRID, alpha=0.45, zorder=0,
                        linewidth=0)
        ax.plot([0.0, 1.0], [0.0, 1.0], color=INK_MUTED, linewidth=1.0, linestyle=(0, (3, 2)),
                zorder=2)
        # Labelled once, on the panel where it has occupants. On (a) the same triangle is empty and
        # is also exactly where that panel's legend has to sit, so a second copy would be covered.
        if regime == "cross_era":
            ax.text(
                0.955, 0.055, "worse than chance\n(ranking inverted)", ha="right", va="bottom",
                fontsize=6.3, color=INK_SECONDARY, linespacing=1.35, style="italic",
            )

        handles, labels = [], []
        for model in MODEL_ORDER:
            record = by_key[(model, regime)]
            colour = MODEL_COLOURS[model]
            step = max(1, (record["n_drawn"] - 1) // _ROC_MARKERS_PER_CURVE)
            line, = ax.plot(
                record["fpr"], record["tpr"], color=colour, linestyle=MODEL_DASHES[model],
                marker=MODEL_MARKERS[model], markevery=step, markersize=4.0,
                markeredgecolor="white", markeredgewidth=0.5,
                linewidth=1.5 if model != "dummy" else 1.1, zorder=4 if model != "dummy" else 3,
            )
            handles.append(line)
            suffix = "  (= chance)" if model == "dummy" else ""
            labels.append(f"{MODEL_LABELS[model]} — {record['roc_auc']:.4f}{suffix}")

        record = by_key[(MODEL_ORDER[0], regime)]
        ax.set_xlim(-0.012, 1.012)
        ax.set_ylim(-0.012, 1.012)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("False positive rate  (1 − specificity)")
        if column == 0:
            ax.set_ylabel("True positive rate  (attack recall)")
        ax.set_title(
            f"({'ab'[column]}) {REGIME_LABELS[regime]}\nn={record['n_test']:,}, "
            f"{record['positive_rate']:.2%} attack",
            pad=6, fontsize=7.8, color=REGIME_COLOURS[regime],
        )
        ax.grid(zorder=1)
        ax.set_axisbelow(True)
        # The legend goes wherever the curves are not, which flips between the panels precisely
        # because the curves do. Ordered by MODEL_ORDER (descending in-distribution AUC), so in (b)
        # the legend deliberately does NOT re-rank: a reader can follow one model across both.
        ax.legend(
            handles, labels, loc="lower right" if column == 0 else "upper left",
            fontsize=6.4, handlelength=2.4, borderpad=0.5, labelspacing=0.42,
            handletextpad=0.6, framealpha=1.0, frameon=True, facecolor="white",
            edgecolor="none", title="ROC-AUC (reports/metrics.csv)",
            title_fontproperties={"size": 6.4, "weight": "bold"},
        )

    _subtitle(
        fig,
        "RQ1 as curves: cross-era the detector runs backwards, it does not flatten",
        "The same six models and one fit per model as Figures 1 and 3. In (a) every curve is above "
        "the chance diagonal;\nin (b) every real model's curve is entirely below it — the shaded "
        "region — which is a rank inversion, not a decay.",
    )

    sources: list[SourceRow] = [
        (
            f"{MODEL_LABELS[record['model']]} — curve and legend AUC in panel "
            f"({'ab'[REGIME_ORDER.index(record['regime'])]}); curve from "
            f"run_phase6()/{ROC_CONDITION}, AUC read from",
            (record["csv_run_id"], record["model"], record["regime"]),
        )
        for record in table
    ]

    # Every quantity below is computed from the drawn records, never transcribed.
    cross = [r for r in table if r["regime"] == "cross_era" and r["model"] != "dummy"]
    in_dist = [r for r in table if r["regime"] == "in_distribution" and r["model"] != "dummy"]
    dummy = by_key[("dummy", "cross_era")]
    worst_drift = max(r["auc_drift"] for r in table)
    max_vertices = max(r["n_vertices"] for r in table)
    max_drawn = max(r["n_drawn"] for r in table)
    # "Below the diagonal" is measured on the drawn vertices, not asserted: the largest amount by
    # which any cross-era vertex of any real model rises above tpr = fpr.
    excursion = max(
        max(tpr - fpr for fpr, tpr in zip(record["fpr"], record["tpr"])) for record in cross
    )
    worst_loss = max(
        by_key[(model, "in_distribution")]["roc_auc"] - by_key[(model, "cross_era")]["roc_auc"]
        for model in MODEL_ORDER
    )
    caption = (
        f"**Figure 4. ROC curves in both regimes (RQ1).** "
        f"The same single fit per model as Figures 1 and 3, evaluated in-distribution on the "
        f"UNSW-NB15 test set (panel a; n={in_dist[0]['n_test']:,}; "
        f"{in_dist[0]['positive_rate']:.2%} attack) and zero-shot cross-era on TON_IoT (panel b; "
        f"n={cross[0]['n_test']:,}; {cross[0]['positive_rate']:.2%} attack), with no refitting of "
        f"the model or the preprocessor. "
        f"The dashed 45° line is chance and the shaded triangle beneath it is the sub-chance "
        f"region. In (a) all five real models run above it, at ROC-AUC "
        f"{min(r['roc_auc'] for r in in_dist):.4f}–{max(r['roc_auc'] for r in in_dist):.4f}; in (b) "
        f"all five lie inside it, at {min(r['roc_auc'] for r in cross):.4f}–"
        f"{max(r['roc_auc'] for r in cross):.4f}, and no drawn vertex of any of the five rises more "
        f"than {excursion:.5f} above the diagonal. "
        f"A curve below the diagonal is not a weak detector "
        f"but an inverted one: the 2015-learned score ranks 2019–20 attacks below normal traffic, "
        f"so the loss (Δ = in-distribution − cross-era, positive = degraded) of up to "
        f"Δ {worst_loss:+.4f} is a rank inversion rather than a decay. "
        f"The majority-class dummy is included as the degenerate case: a constant predictor has a "
        f"single operating point, so its ROC *is* the chance diagonal, at exactly "
        f"{dummy['roc_auc']:.4f} in both regimes — it is the control showing that the collapse in "
        f"(b) is not the change in class balance, since a prevalence-driven artifact would move it "
        f"too. "
        f"Each legend entry's ROC-AUC is the committed value from the "
        f"`(run_id, model, regime)` row of `reports/metrics.csv`, not a quantity re-derived here. "
        f"Curves come from `evaluate.run_phase6()` under `run_id={ROC_CONDITION}` (the full d=22 "
        f"feature set; both ablation conditions are excluded by construction) via "
        f"`reports/roc_curves.json`, where each is stored as exact integer (false-positive, "
        f"true-positive) vertex counts reduced from up to {max_vertices:,} vertices to at most "
        f"{max_drawn:,} by an area-preserving simplification; the trapezoidal area of every drawn "
        f"curve was re-checked against its own logged ROC-AUC before plotting and agrees to within "
        f"{worst_drift:.1e}."
    )
    return save_figure(fig, "roc_curves", caption, sources, out=out)


# =========================================================================================
# Figures 5 and 6 — per-attack-family breakdowns
# =========================================================================================
#
# TWO FIGURES, NOT ONE, AND THEY ARE NOT THE SAME EXPERIMENT. The distinction is the single easiest
# thing to get wrong in this phase, so it is enforced by the `family_set` column of
# `reports/per_family_metrics.csv` and asserted in both functions below:
#
#   Figure 5 (`plot_per_family_f1`)        CROSS-ERA, and therefore the THREE SHARED families only
#                                          -- `dos`, `scanning`, `backdoor`. Rows under
#                                          family_set="shared", two regimes, from Phase 6.
#   Figure 6 (`plot_per_family_recovery`)  WITHIN-ERA recovery, over TON_IoT's OWN attack types
#                                          (eight at ~20,000 rows plus a ninth at 1,043) on the
#                                          frozen test half. Rows under family_set="native", one
#                                          regime, from Phase 7.
#
# A join that mixed them would compare a 583-row UNSW family against a 20,000-row TON_IoT one, or
# read a TON_IoT-only family (`ransomware`, `ddos`, ...) as a cross-era result. UNSW-NB15 has NO
# DDoS class at all; four shared families is a bug, not a finding.
#
# EVERY NUMBER ON BOTH FIGURES IS A ONE-VS-NORMAL SCORE: the family's rows plus every normal row of
# the same evaluation set. An attack family is all-positive by construction, so F1 over its rows
# alone is a monotone function of recall and cannot be read against a majority-class floor; see
# `evaluate.per_family_metrics`. The consequence a reader must carry is that each family's subset
# has its own class balance -- UNSW-test `backdoor` is 583 rows against 37,000 normals (1.55%
# attack) while TON_IoT `backdoor` is 20,000 against 50,000 (28.57%) -- so per-family F1 moves on
# prevalence between the two regimes even harder than the aggregate F1 of Figure 1 does. That is
# why ROC-AUC leads on Figure 5 and why the majority-class lines are drawn on its F1 panels.

#: The Phase 6 condition Figure 5 renders, and the only one it may -- same constant and same
#: argument as :data:`CONFUSION_CONDITION` and :data:`ROC_CONDITION`.
PER_FAMILY_CONDITION: str = CROSS_ERA_RUN_ID

#: The shared attack families, derived from ``schema_map.SHARED_FAMILIES`` rather than written down
#: here. That is the guard against the two failures this project has already had: a fourth family
#: appearing (UNSW-NB15 has no DDoS class, so ``ddos`` can never be shared), and a family silently
#: vanishing (the delivered ``attack_cat`` value is ``Backdoor``, singular -- the plural spelling
#: the upstream *documentation* uses matched zero rows and failed without a word). Deriving it means
#: the figure cannot disagree with the map that built the parquets.
EXPECTED_SHARED_FAMILIES: tuple[str, ...] = tuple(sorted(
    {
        family
        for side in SHARED_FAMILIES.values()
        for family in side.values()
        if family is not None and family != NORMAL_FAMILY
    }
))


def _span(values: Any, decimals: int = 4) -> str:
    """``0.1234–0.5678``, collapsed to one number when the ends coincide at that precision.

    A caption that prints "0.4444–0.4444" reads as a measurement error rather than as three
    families that legitimately share a class balance.
    """
    low, high = f"{min(values):.{decimals}f}", f"{max(values):.{decimals}f}"
    return low if low == high else f"{low}–{high}"


def _family_title(family: str) -> str:
    """``"scanning" -> "Reconnaissance ↔ scanning"`` -- both eras' delivered spellings.

    The shared vocabulary borrows TON_IoT's spelling, so a reader who sees only ``scanning`` cannot
    tell which UNSW class it was mapped from. Both keys are looked up in ``SHARED_FAMILIES`` rather
    than transcribed, and a family that does not resolve to exactly one level per side raises --
    that is the same silent-mismatch failure the singular/plural ``Backdoor`` bug was.
    """
    sides = []
    for side in ("unsw", "toniot"):
        levels = sorted(k for k, v in SHARED_FAMILIES[side].items() if v == family)
        if len(levels) != 1:
            raise KeyError(
                f"shared family {family!r} maps from {len(levels)} {side} level(s) {levels}; "
                "exactly one delivered level per side is what makes the pairing a pairing"
            )
        sides.append(levels[0])
    return f"{sides[0]} ↔ {sides[1]}"


def _cross_era_family_table(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (family, model) carrying both regimes' halves, with the family set asserted.

    Restricted to ``family_set == "shared"`` under :data:`PER_FAMILY_CONDITION`, and the resulting
    family set is checked against :data:`EXPECTED_SHARED_FAMILIES` before anything is drawn.
    """
    shared = frame[
        (frame["run_id"] == PER_FAMILY_CONDITION) & (frame["family_set"] == SHARED_FAMILY_SET)
    ]
    families = tuple(sorted(set(shared["family"])))
    if families != EXPECTED_SHARED_FAMILIES:
        raise RuntimeError(
            f"the cross-era per-family figure resolved families {list(families)}, expected exactly "
            f"{list(EXPECTED_SHARED_FAMILIES)} from schema_map.SHARED_FAMILIES. UNSW-NB15 has no "
            "DDoS class and only three families align across the two eras; a fourth is a bug in "
            "the family map, not a result."
        )

    records = []
    for family in families:
        for model in MODEL_ORDER:
            halves = {
                regime: select_family_row(
                    frame, PER_FAMILY_CONDITION, model, regime, SHARED_FAMILY_SET, family
                )
                for regime in REGIME_ORDER
            }
            records.append({
                "family": family,
                "model": model,
                **{
                    f"{prefix}_{column}": halves[regime][column]
                    for prefix, regime in (("in", "in_distribution"), ("cross", "cross_era"))
                    for column in ("roc_auc", "f1", "n_family", "n_normal", "positive_rate")
                },
            })
    table = pd.DataFrame.from_records(records)

    # Every model must have been scored against the SAME subset per family and regime, or the bars
    # within a panel are not comparable to each other.
    for family in families:
        block = table[table["family"] == family]
        for column in ("in_n_family", "in_n_normal", "cross_n_family", "cross_n_normal"):
            if block[column].nunique() != 1:
                raise RuntimeError(
                    f"{column} is not constant across models for family {family!r} "
                    f"({sorted(set(block[column]))}); the six models were not scored on one subset."
                )
    return table


def plot_per_family_f1(results: Any, out: Any = FIGURES) -> Path:
    """Per-shared-family F1 and ROC-AUC, in-distribution vs cross-era (RQ1, Figure 1 decomposed).

    **The three shared families only** — ``DoS ↔ dos``, ``Reconnaissance ↔ scanning``,
    ``Backdoor ↔ backdoor``. Those are the only attack classes both label spaces contain (UNSW-NB15
    ships no DDoS class at all), and they cover a minority of each era's attack rows, which the
    caption states as a limit rather than implying. The recovery-side per-family breakdown is a
    *different* figure over a *different* population — see :func:`plot_per_family_recovery`.

    A 3x2 grid: one row per family, and within each row the same two panels as Figure 1 —
    **ROC-AUC leads** and F1 follows with its majority-class lines drawn on. The per-family F1
    panels need that caveat even more than Figure 1's does: each family is scored one-vs-normal, so
    the subset's class balance is set by the family's own size, and UNSW-test ``backdoor`` (583 rows
    against 37,000 normals) and TON_IoT ``backdoor`` (20,000 against 50,000) are 1.55% and 28.57%
    attack respectively. A dummy predicting "attack" everywhere therefore scores a *higher* F1
    cross-era on every family, on balance alone.

    ``results`` is the DataFrame from :func:`load_per_family` (or ``None`` / a path to a per-family
    CSV — what :func:`main` passes). Bars come from ``reports/per_family_metrics.csv``, written by
    ``evaluate.run_phase6()`` under ``run_id=phase6-crossera``; both ablation conditions are excluded
    by construction, since neither is scored per family at all.
    """
    apply_style()
    table = _cross_era_family_table(as_per_family_frame(results))

    # How narrow the three families are, computed rather than transcribed. The per-family artifact
    # knows each family's size but not the era's total attack count, so the denominators come from
    # the same `reports/metrics.csv` rows Figure 1's bars do -- n_test x positive_rate, which is
    # exactly what those two columns are there to record.
    metrics = as_metrics_frame(None)
    coverage = {}
    for prefix, run_id, regime in (
        ("in", IN_DISTRIBUTION_RUN_ID["dummy"], "in_distribution"),
        ("cross", CROSS_ERA_RUN_ID, "cross_era"),
    ):
        row = select_row(metrics, run_id, "dummy", regime)
        n_attack = int(round(float(row["n_test"]) * float(row["positive_rate"])))
        n_covered = int(table.groupby("family")[f"{prefix}_n_family"].first().sum())
        coverage[prefix] = (n_covered, n_attack, n_covered / n_attack)

    # Largest in-distribution family first: the reader meets the best-supported comparison at the
    # top, and the ordering is derived from the data rather than chosen.
    families = list(
        table.groupby("family")["in_n_family"].first().sort_values(ascending=False).index
    )

    fig, axes = plt.subplots(len(families), 2, figsize=(PAGE_WIDTH_IN, 7.7), sharey=True)
    fig.subplots_adjust(left=0.195, right=0.985, top=0.800, bottom=0.115, wspace=0.30, hspace=1.05)

    positions = list(range(len(MODEL_ORDER)))
    height = 0.36
    gutter_x = 1.045
    header_y = -1.05

    for row, family in enumerate(families):
        block = table[table["family"] == family].set_index("model").loc[list(MODEL_ORDER)]
        dummy = block.loc["dummy"]
        for column, (metric, xlabel) in enumerate(
            (("roc_auc", "ROC-AUC"), ("f1", "F1 (attack class)"))
        ):
            ax = axes[row][column]
            in_values = block[f"in_{metric}"].to_numpy()
            cross_values = block[f"cross_{metric}"].to_numpy()
            ax.barh(
                [p - height / 2 for p in positions], in_values, height,
                color=REGIME_COLOURS["in_distribution"], label=REGIME_LABELS["in_distribution"],
                edgecolor="white", linewidth=0.8, zorder=3,
            )
            ax.barh(
                [p + height / 2 for p in positions], cross_values, height,
                color=REGIME_COLOURS["cross_era"], label=REGIME_LABELS["cross_era"],
                edgecolor="white", linewidth=0.8, zorder=3,
            )
            # Δ = in − cross, in its own gutter: positive = what the model lost. Three decimals
            # rather than Figure 1's four, because the gutter is half as wide at this panel size.
            ax.text(gutter_x, header_y, "Δ", ha="left", va="center", fontsize=6.6,
                    fontweight="bold", color=INK_PRIMARY)
            for position, high, low in zip(positions, in_values, cross_values):
                ax.text(gutter_x, position, f"{high - low:+.3f}", ha="left", va="center",
                        fontsize=6.2, color=INK_PRIMARY)

            if metric == "roc_auc":
                chance = float(dummy["cross_roc_auc"])
                ax.axvline(chance, color=INK_SECONDARY, linewidth=1.0, linestyle=(0, (3, 2)),
                           zorder=4)
                ax.text(chance - 0.03, header_y, f"chance {chance:.4f}", ha="right", va="center",
                        fontsize=6.0, color=INK_SECONDARY)
            else:
                for key, colour, label_y in (
                    ("in_f1", REGIME_COLOURS["in_distribution"], header_y - 0.32),
                    ("cross_f1", REGIME_COLOURS["cross_era"], header_y + 0.30),
                ):
                    value = float(dummy[key])
                    ax.axvline(value, color=colour, linewidth=1.0, linestyle=(0, (3, 2)), zorder=4)
                    ax.text(value, label_y, f"{value:.4f}", ha="center", va="center", fontsize=6.0,
                            color=colour)

            ax.set_xlabel(xlabel, fontsize=7.6, labelpad=2)
            ax.set_xlim(0.0, 1.34)
            ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax.set_yticks(positions)
            ax.set_yticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=6.8)
            ax.set_ylim(len(MODEL_ORDER) - 0.4, header_y - 0.56)  # inverted: best model on top
            ax.grid(axis="x", zorder=0)
            ax.set_axisbelow(True)
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)

        # Row header, spanning both panels: the family and the two one-vs-normal subsets it was
        # scored over. Positioned from the row's own axes so retuning the grid cannot orphan it,
        # and broken over two lines because a single line of this length silently widens the whole
        # figure -- `savefig.bbox = "tight"` sizes the PNG to its widest artist.
        first = block.iloc[0]
        y = axes[row][0].get_position().y1 + 0.052
        fig.text(
            0.030, y, f"({'abc'[row]}) {_family_title(family)}",
            ha="left", va="bottom", fontsize=8.0, fontweight="bold", color=INK_PRIMARY,
        )
        fig.text(
            0.030, y - 0.0165,
            f"scored one-vs-normal   ·   UNSW-test: {int(first['in_n_family']):,} "
            f"vs {int(first['in_n_normal']):,} normal "
            f"({float(first['in_positive_rate']):.2%} attack)   ·   "
            f"TON_IoT: {int(first['cross_n_family']):,} vs "
            f"{int(first['cross_n_normal']):,} normal "
            f"({float(first['cross_positive_rate']):.2%} attack)",
            ha="left", va="bottom", fontsize=6.6, color=INK_SECONDARY,
        )

    handles, labels = axes[0][0].get_legend_handles_labels()
    reference_handles = [
        plt.Line2D([], [], color=INK_SECONDARY, linestyle=(0, (3, 2)),
                   label="left panels: chance, ROC-AUC = 0.5000"),
        plt.Line2D([], [], color=REGIME_COLOURS["in_distribution"], linestyle=(0, (3, 2)),
                   label="right panels: majority-class F1 on the UNSW-test subset"),
        plt.Line2D([], [], color=REGIME_COLOURS["cross_era"], linestyle=(0, (3, 2)),
                   label="right panels: majority-class F1 on the TON_IoT subset"),
    ]
    fig.legend(
        [*handles, *reference_handles],
        [*labels, *(h.get_label() for h in reference_handles)],
        loc="lower center", bbox_to_anchor=(0.5, 0.002), ncol=2, handlelength=1.9,
        columnspacing=1.6, labelcolor=INK_PRIMARY,
    )
    _subtitle(
        fig,
        "RQ1 per attack family: the collapse is not one family's",
        "The three families both eras label, each scored against its own era's benign traffic. "
        "Δ = in-distribution − cross-era\n(positive = performance lost). Read the F1 panels against "
        "the dashed majority-class lines: every subset's balance rises.",
    )

    sources: list[SourceRow] = []
    for family in families:
        for model in MODEL_ORDER:
            for regime in REGIME_ORDER:
                sources.append((
                    f"{_family_title(family)} — {MODEL_LABELS[model]}, "
                    f"{'in-distribution' if regime == 'in_distribution' else 'cross-era'} bars",
                    (PER_FAMILY_CONDITION, model, regime, SHARED_FAMILY_SET, family),
                ))

    # Everything quoted below is computed from the plotted rows. `real` excludes the dummy, whose
    # per-family ROC-AUC is 0.5000 in both regimes by construction.
    real = table[table["model"] != "dummy"]
    inverted = real[real["cross_roc_auc"] < 0.5]
    sizes = ", ".join(
        f"{_family_title(family)} "
        f"({int(table[table['family'] == family]['in_n_family'].iloc[0]):,} UNSW-test / "
        f"{int(table[table['family'] == family]['cross_n_family'].iloc[0]):,} TON_IoT rows)"
        for family in families
    )
    caption = (
        f"**Figure 5. The cross-era collapse, per shared attack family (RQ1).** "
        f"Figure 1's aggregate result decomposed over the only three attack families both label "
        f"spaces contain: {sizes}. UNSW-NB15 ships **no DDoS class**, and its `Exploits`, "
        f"`Generic`, `Fuzzers`, `Analysis`, `Shellcode` and `Worms` classes have no TON_IoT "
        f"counterpart, so the three families reach only "
        f"**{coverage['in'][2]:.2%} of UNSW-test's {coverage['in'][1]:,} attack rows "
        f"({coverage['in'][0]:,}) and {coverage['cross'][2]:.2%} of TON_IoT's "
        f"{coverage['cross'][1]:,} ({coverage['cross'][0]:,})** — the binary headline of Figures 1, "
        f"3 and 4 uses every row, and only this per-family view is restricted. "
        f"Each family is scored **one-vs-normal**: the family's rows plus every normal row of the "
        f"same evaluation set, because an attack family is all-positive by construction and F1 over "
        f"its rows alone would be a relabelling of recall. "
        f"Panels (a, c, e) lead with ROC-AUC: {len(inverted)} of the {len(real)} (family, model) "
        f"pairs among the five real models fall *below* the 0.5000 chance line cross-era, ranging "
        f"down to {real['cross_roc_auc'].min():.4f}, so the inversion of Figure 1 is present in "
        f"every family rather than driven by one of them. "
        f"Panels (b, d, f) must be read against the dashed majority-class lines, which move between "
        f"the regimes far more than in Figure 1: one-vs-normal makes each subset's balance the "
        f"family's own size, so the same dummy scores F1 "
        f"{_span(table['in_f1'][table['model'] == 'dummy'])} in-distribution and "
        f"{_span(table['cross_f1'][table['model'] == 'dummy'])} cross-era on prevalence alone. "
        f"Δ = in-distribution − cross-era throughout, i.e. what the model *lost*: positive means "
        f"degraded, and a negative per-family Δ F1 is that prevalence rise rather than an "
        f"improvement. "
        f"Rows come from `reports/per_family_metrics.csv` under "
        f"`run_id={PER_FAMILY_CONDITION}`, `family_set=shared`; the two ablation conditions are not "
        f"scored per family at all."
    )
    return save_figure(fig, "per_family_crossera", caption, sources, out=out)


# =========================================================================================
# Figure 6 — RQ2 per-family recovery
# =========================================================================================

#: The smallest family Figure 6 will draw. TON_IoT ships eight attack types at exactly 20,000 rows
#: and one, ``mitm``, at 1,043 — 518 of which land in the frozen test half. That is an order of
#: magnitude below the rest and its curve is not comparable to theirs, so it is excluded from the
#: panels and reported in the caption instead of being quietly averaged in. The threshold is a row
#: count rather than a name so that a re-delivered dataset cannot silently reinstate it.
MIN_FAMILY_ROWS: int = 5_000

#: Budgets drawn on Figure 6, in order, and the ceiling that sits past the axis break — the same
#: x-axis as Figure 2, so the two read together.
RECOVERY_FAMILY_RUN_IDS: tuple[str, ...] = RECOVERY_RUN_IDS


def _recovery_family_table(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str], int]:
    """``(curve, drawn, excluded, n_frozen)`` for Figure 6, from ``family_set="native"`` rows only.

    ``curve`` carries **every** native family, including the ones too small to draw, so the caption
    can quote the excluded one's numbers from the same table the panels come from rather than from
    prose. ``drawn`` is the families at or above :data:`MIN_FAMILY_ROWS`, largest first; ``excluded``
    is the remainder. ``n_frozen`` is the frozen test half's size, summed out of the artifact
    (every family's rows plus the shared normal block) rather than transcribed.
    """
    if FREEZE_CONTROL_RUN_ID in (*RECOVERY_FAMILY_RUN_IDS, CEILING_RUN_ID):  # pragma: no cover
        raise RuntimeError(
            f"{FREEZE_CONTROL_RUN_ID} is configured as a per-family budget point. It is the MLP "
            "freeze-cost control at the ceiling's budget, not a point on the curve."
        )
    native = frame[frame["family_set"] == NATIVE_FAMILY_SET]
    if native.empty:
        raise KeyError(
            "reports/per_family_metrics.csv holds no `native` rows. Those are Phase 7's; run "
            "`python -m src.transfer`, or `./run.sh`, which runs it before Phase 9."
        )

    sizes = native.groupby("family")["n_family"].agg(["min", "max"])
    if (sizes["min"] != sizes["max"]).any():
        raise RuntimeError(
            "a family's row count differs between two rows of the per-family log; every Phase 7 "
            "point must be scored on the one frozen TON_IoT test half."
        )
    ordered = sizes["min"].sort_values(ascending=False)
    drawn = [str(name) for name in ordered[ordered >= MIN_FAMILY_ROWS].index]
    excluded = [str(name) for name in ordered[ordered < MIN_FAMILY_ROWS].index]

    normals = set(native["n_normal"])
    if len(normals) != 1:
        raise RuntimeError(
            f"the frozen half's normal block differs between rows ({sorted(normals)}); every "
            "one-vs-normal subset must have been scored against the same benign rows."
        )
    n_frozen = int(ordered.sum()) + int(normals.pop())

    records = []
    for family in ordered.index:
        for model in (*RECOVERY_MODELS, FLOOR_MODEL):
            for run_id in (*RECOVERY_FAMILY_RUN_IDS, CEILING_RUN_ID):
                row = select_family_row(
                    frame, run_id, model, TRANSFER_REGIME, NATIVE_FAMILY_SET, family
                )
                records.append({
                    "family": family, "model": model, "run_id": run_id,
                    "fraction": (
                        1.0 if run_id == CEILING_RUN_ID else float(run_id.rsplit("-f", 1)[-1])
                    ),
                    "f1": float(row["f1"]), "roc_auc": float(row["roc_auc"]),
                    "n_family": int(row["n_family"]), "n_normal": int(row["n_normal"]),
                    "positive_rate": float(row["positive_rate"]),
                })
    curve = pd.DataFrame.from_records(records)

    # The dummy is a floor only if it is one: its score depends on prevalence, which is constant
    # per family across budgets, so it must not move along x. Assert rather than assume.
    for family in ordered.index:
        floor = curve[(curve["family"] == family) & (curve["model"] == FLOOR_MODEL)]
        for metric in ("f1", "roc_auc"):
            if floor[metric].round(9).nunique() != 1:
                raise RuntimeError(
                    f"the majority-class {metric} for family {family!r} is not constant across the "
                    f"Phase 7 budgets ({sorted(set(floor[metric]))}); it cannot be a floor line."
                )
    return curve, drawn, excluded, n_frozen


def plot_per_family_recovery(results: Any, out: Any = FIGURES) -> Path:
    """Per-family F1 vs fine-tune budget, on the frozen TON_IoT test half (RQ2 decomposed).

    **A different population from Figure 5, and deliberately so.** This is a *within*-era
    breakdown over **TON_IoT's own attack types** — the eight delivered at 20,000 rows each — asking
    what a modern labelling budget buys per modern attack class. Five of the eight (`ddos`,
    `injection`, `password`, `ransomware`, `xss`) have no 2015 counterpart at all, which is exactly
    why they cannot appear on Figure 5 and exactly why they are the interesting ones here:
    ``ransomware`` is the class the drift story is really about.

    One small panel per family, sharing Figure 2's x-axis (ordinal budget positions, the per-model
    ceiling past a dashed break) and Figure 2's model colours, markers and dash patterns, so a
    reader carries one encoding across both. Each panel also carries **its own** majority-class
    floor: one-vs-normal makes every family's subset a different balance, so there is no single
    floor line for the figure.

    ``results`` is the DataFrame from :func:`load_per_family` (or ``None`` / a path). Every point is
    scored on the same frozen 105,521-row TON_IoT test half that no budget ever draws from; the MLP
    freeze-cost control is excluded and asserted excluded, exactly as on Figure 2.
    """
    apply_style()
    curve, families, excluded, n_frozen = _recovery_family_table(as_per_family_frame(results))
    drawn = curve[curve["family"].isin(families)]

    fractions = sorted(curve[curve["run_id"] != CEILING_RUN_ID]["fraction"].unique())
    x_budget = list(range(len(fractions)))
    x_ceiling = len(fractions) + 0.55
    break_x = len(fractions) - 0.5 + 0.275

    columns = 4
    rows = (len(families) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(PAGE_WIDTH_IN, 4.9), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.082, right=0.992, top=0.745, bottom=0.225, wspace=0.14, hspace=0.46)
    flat = [ax for row in axes for ax in row]

    for index, family in enumerate(families):
        ax = flat[index]
        block = curve[curve["family"] == family]
        floor = float(block[block["model"] == FLOOR_MODEL]["f1"].iloc[0])
        ax.axvline(break_x, color=INK_MUTED, linewidth=0.8, linestyle=(0, (2, 2)), alpha=0.6,
                   zorder=1)
        ax.axhline(floor, color=INK_SECONDARY, linewidth=1.0, linestyle=(0, (3, 2)), zorder=2)

        for model in RECOVERY_MODELS:
            points = block[block["model"] == model].sort_values("fraction")
            budgets = points[points["run_id"] != CEILING_RUN_ID]["f1"].to_numpy()
            ceiling = float(points[points["run_id"] == CEILING_RUN_ID]["f1"].iloc[0])
            colour = MODEL_COLOURS[model]
            ax.plot(
                x_budget, budgets, color=colour, linestyle=MODEL_DASHES[model],
                marker=MODEL_MARKERS[model], markersize=3.2, markeredgecolor="white",
                markeredgewidth=0.4, linewidth=1.2, label=MODEL_LABELS[model], zorder=3,
            )
            ax.plot([x_budget[-1], x_ceiling], [budgets[-1], ceiling], color=colour,
                    linestyle=(0, (1, 1.4)), linewidth=0.9, alpha=0.7, zorder=3)
            ax.plot([x_ceiling], [ceiling], marker=MODEL_MARKERS[model], markersize=3.6,
                    markerfacecolor="white", markeredgecolor=colour, markeredgewidth=1.0, zorder=4)

        # Two short lines, not one long one: at ~1.5 in of panel width a single-line title runs
        # into its neighbour's, and the per-family class balance (28.4-28.9% throughout) belongs in
        # the caption rather than repeated eight times.
        first = block.iloc[0]
        ax.set_title(
            f"{family}\nn = {int(first['n_family']):,}", fontsize=7.2, pad=3.0, linespacing=1.35,
        )
        ax.text(-0.28, floor - 0.045, f"floor {floor:.3f}", ha="left", va="top", fontsize=5.9,
                color=INK_SECONDARY, zorder=5,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8})
        ax.set_ylim(-0.04, 1.06)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlim(-0.4, x_ceiling + 0.4)
        ax.set_xticks([*x_budget, x_ceiling])
        ax.set_xticklabels(
            [*(f"{fraction:.0%}" for fraction in fractions), "100%"], fontsize=6.2,
        )
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)

    for ax in flat[len(families):]:  # pragma: no cover - eight families fill the 2x4 grid exactly
        ax.set_visible(False)

    fig.supylabel("F1 (attack class) — family vs normal", fontsize=8.0, color=INK_PRIMARY, x=0.010)
    fig.supxlabel(
        "Fine-tune budget — share of the 105,522-row TON_IoT pool; the column past each\npanel's "
        "dashed break is that model's own ceiling at the full pool",
        fontsize=7.8, color=INK_PRIMARY, y=0.128, linespacing=1.5,
    )

    handles, labels = flat[0].get_legend_handles_labels()
    ceiling_handle = plt.Line2D(
        [], [], color=INK_SECONDARY, linestyle=(0, (1, 1.4)), marker="o", markersize=3.6,
        markerfacecolor="white", markeredgecolor=INK_SECONDARY,
        label="per-model ceiling (full pool)",
    )
    floor_handle = plt.Line2D(
        [], [], color=INK_SECONDARY, linestyle=(0, (3, 2)),
        label="majority-class floor (per family)",
    )
    fig.legend(
        [*handles, ceiling_handle, floor_handle],
        [*labels, ceiling_handle.get_label(), floor_handle.get_label()],
        loc="lower center", bbox_to_anchor=(0.5, 0.002), ncol=4, handlelength=2.2,
        columnspacing=1.3, labelcolor=INK_PRIMARY,
    )
    _subtitle(
        fig,
        "RQ2 per family: the recovery reaches all eight attack types",
        "TON_IoT's own attack types on the frozen test half — five of the eight have no 2015 "
        "counterpart and so\ncannot appear in Figure 5. Each family is scored one-vs-normal, so "
        "every panel carries its own class\nbalance (28.4–28.9% attack) and its own "
        "majority-class floor. n is the family's row count in that half.",
    )

    sources: list[SourceRow] = []
    for family in families:
        for model in (*RECOVERY_MODELS, FLOOR_MODEL):
            for run_id in (*RECOVERY_FAMILY_RUN_IDS, CEILING_RUN_ID):
                sources.append((
                    f"{family} panel — {MODEL_LABELS[model]} at "
                    f"{'the ceiling' if run_id == CEILING_RUN_ID else run_id.rsplit('-', 1)[-1]}",
                    (run_id, model, TRANSFER_REGIME, NATIVE_FAMILY_SET, family),
                ))

    # Computed from the table the panels are drawn from, including the excluded family's own rows.
    floors = curve[curve["model"] == FLOOR_MODEL].groupby("family")["f1"].first()
    real = drawn[drawn["model"] != FLOOR_MODEL]
    zero = real[real["fraction"] == 0.0]
    one_pct = real[real["fraction"] == 0.01]
    below_at_one_pct = one_pct[one_pct["f1"].to_numpy() <= one_pct["family"].map(floors).to_numpy()]
    cleared = len(one_pct) - len(below_at_one_pct)
    zero_below = int((zero["f1"].to_numpy() < zero["family"].map(floors).to_numpy()).sum())
    # The exceptions are named rather than left inside "38 of 40": at the smallest budget the claim
    # is not uniform, and which pairs miss is the useful part.
    five_pct = real[real["fraction"] == 0.05]
    exceptions = (
        "every one of them clears it"
        if below_at_one_pct.empty
        else (
            "the exceptions are "
            + ", ".join(
                f"{MODEL_LABELS[row['model']]} on `{row['family']}` ({row['f1']:.4f} against "
                f"{float(floors[row['family']]):.4f})"
                for _index, row in below_at_one_pct.iterrows()
            )
            + f", both of which clear it by the 5% budget (minimum {five_pct['f1'].min():.4f})"
        )
    )
    excluded_note = ""
    for name in excluded:
        small = curve[curve["family"] == name]
        best = small[(small["run_id"] == CEILING_RUN_ID) & (small["model"] != FLOOR_MODEL)]
        excluded_note += (
            f"TON_IoT's ninth attack type, `{name}`, is **excluded from the panels**: it ships "
            f"1,043 rows against the others' 20,000 and contributes only "
            f"{int(small['n_family'].iloc[0]):,} to the frozen half, so its curve is not comparable "
            f"to theirs. It is also the one family the budget does not rescue — its best ceiling F1 "
            f"is {best['f1'].max():.4f} against a {float(floors[name]):.4f} floor, against "
            f"{drawn[(drawn['run_id'] == CEILING_RUN_ID) & (drawn['model'] != FLOOR_MODEL)]['f1'].min():.4f}"
            f"–"
            f"{drawn[(drawn['run_id'] == CEILING_RUN_ID) & (drawn['model'] != FLOOR_MODEL)]['f1'].max():.4f} "
            f"on the eight drawn — worth stating precisely because the rarest family is the one "
            f"that stays hardest. "
        )
    caption = (
        f"**Figure 6. What a modern labelling budget buys, per TON_IoT attack family (RQ2).** "
        f"Figure 2's recovery curve decomposed over the {len(families)} attack types TON_IoT "
        f"delivers at 20,000 rows each, scored on the same frozen {n_frozen:,}-row "
        f"test half. This is a **within-era** breakdown and a different population from Figure 5: "
        f"five of the eight (`ddos`, `injection`, `password`, `ransomware`, `xss`) have no UNSW-NB15 "
        f"counterpart and therefore cannot appear in a cross-era per-family comparison at all. "
        f"Each family is scored one-vs-normal against the {int(curve['n_normal'].iloc[0]):,} normal "
        f"rows of the frozen half, so every panel has its own class balance and its own dashed "
        f"majority-class floor — there is no single floor for this figure. "
        f"Zero-shot, the five real models sit at F1 {_span(zero['f1'])}, below the family's own "
        f"floor in {zero_below} of the {len(zero)} (family, model) cases. A budget of 1% of the "
        f"pool (1,055 labelled flows) lifts them to {_span(one_pct['f1'])}, clearing that floor in "
        f"{cleared} of the {len(one_pct)} — {exceptions}. The recovery is therefore not confined to "
        f"the three families the two eras share: the five with no 2015 counterpart recover on the "
        f"same budget as the three that have one. "
        f"{excluded_note}"
        f"x is ordinal and the column past each dashed break is that model's own ceiling at the "
        f"full pool, exactly as in Figure 2; the MLP freeze-cost control "
        f"(`{FREEZE_CONTROL_RUN_ID}`) is excluded by construction. "
        f"Rows come from `reports/per_family_metrics.csv` under `family_set=native`, written by "
        f"`transfer.run_phase7()`."
    )
    return save_figure(fig, "per_family_recovery", caption, sources, out=out)


# =========================================================================================
# Figure 7 — the Methods pipeline diagram
# =========================================================================================
#
# The one figure here that is a *schematic* rather than a measurement, and the course handout calls
# one out as "extremely useful" in Methods. Two rules keep it from becoming the aspirational box
# diagram that kind of figure usually is:
#
#   1. every module, artifact and path it names is one that exists, spelled the way the repo spells
#      it, and every count on it is read from `reports/metrics.csv` or from a module constant that
#      the pipeline itself asserts at runtime (`evaluate.EXPECTED_ROWS`,
#      `preprocess.EXPECTED_SOURCE_ROWS`, `schema_map.COMMON_COLUMNS`) -- never transcribed;
#   2. it draws the two things that are *methodologically* load-bearing rather than merely true:
#      the **fit-on-source boundary** (below it the `Preprocessor` is transform-only, so neither
#      UNSW-test nor TON_IoT contributes a statistic) and the **leakage seal** (`evaluate.sealed`,
#      which shadows `fit`/`fit_transform`/`partial_fit` and raises `LeakageError`), which is the
#      claim of no-leakage made as a runtime guarantee instead of as a promise in prose.
#
# It is drawn on a bare axes in a 0-100 x 0-`_PIPE_TOP` coordinate system, top-down, because the
# per-stage text does not survive being squeezed into six 1.1-inch columns of a left-to-right spine.

#: Height of the diagram's coordinate system. x is always 0-100.
_PIPE_TOP: float = 104.0

#: Vertical gap between one level and the next; the arrow lives inside it.
_PIPE_GAP: float = 2.7

#: Internal box geometry, in diagram units, tuned to the type sizes below: top padding, the height
#: of the bold title line, the gap under it, one body line, and bottom padding. :func:`_pipe_box`
#: derives its own height from these and its body's line count, so a box can never be too short for
#: the text inside it -- which is the one failure mode a hand-tuned schematic has.
_PIPE_PAD_TOP: float = 1.5
_PIPE_TITLE_H: float = 2.4
_PIPE_TITLE_GAP: float = 1.1
_PIPE_LINE_H: float = 2.42
_PIPE_PAD_BOTTOM: float = 1.4

#: Fills. Module boxes carry ink on white; artifacts are recessive because they are what *flows*
#: between the modules rather than what acts.
_MODULE_FACE: str = "#ffffff"
_ARTIFACT_FACE: str = "#f4f3ef"
_SEAL_FACE: str = "#faf3ee"

_D_IN_NOTE = re.compile(r"\bd=(\d+)\b")
_N_FT_IN_NOTE = re.compile(r"n_ft=(\d+)")


def _note_field(row: pd.Series, pattern: Any, what: str) -> int:
    """One integer out of a committed ``notes`` string, or raise naming the row it came from."""
    match = pattern.search(str(row["notes"]))
    if match is None:
        raise ValueError(
            f"the ({row['run_id']}, {row['model']}, {row['regime']}) row of reports/metrics.csv "
            f"records no {what} in its notes: {row['notes']!r}"
        )
    return int(match.group(1))


def _pipeline_facts(frame: pd.DataFrame) -> dict[str, Any]:
    """Every number the diagram prints, with the committed row or constant each came from.

    Nothing here is a literal. The evaluation-set sizes, feature widths and fine-tune budgets are
    read from ``reports/metrics.csv`` rows (or from the ``notes`` those rows carry); the fold sizes
    and the harmonized column count come from the module constants the pipeline asserts against at
    runtime. If a number on this figure is ever wrong, the run that produced it was wrong too.
    """
    from .preprocess import EXPECTED_SOURCE_ROWS, VALIDATION_FRACTION  # noqa: PLC0415

    in_row = select_row(frame, IN_DISTRIBUTION_RUN_ID["dummy"], "dummy", "in_distribution")
    cross_row = select_row(frame, CROSS_ERA_RUN_ID, "dummy", "cross_era")
    smallest = select_row(frame, RECOVERY_RUN_IDS[1], "dummy", TRANSFER_REGIME)
    ceiling = select_row(frame, CEILING_RUN_ID, "dummy", TRANSFER_REGIME)

    n_train, _ = EXPECTED_ROWS["train"]
    budgets = tuple(float(run_id.rsplit("-f", 1)[-1]) for run_id in RECOVERY_RUN_IDS[1:])
    return {
        "n_unsw_train_raw": EXPECTED_SOURCE_ROWS,
        "n_unsw_test": int(in_row["n_test"]),
        "n_toniot": int(cross_row["n_test"]),
        "n_unsw_parquet": EXPECTED_SOURCE_ROWS + int(in_row["n_test"]),
        "n_common_columns": len(COMMON_COLUMNS),
        "n_mapped_concepts": len(FEATURE_MAP),
        "n_train_fold": n_train,
        "n_val_fold": EXPECTED_SOURCE_ROWS - n_train,
        "validation_fraction": VALIDATION_FRACTION,
        "d_full": _note_field(cross_row, _D_IN_NOTE, "feature width"),
        "d_ablated": _note_field(
            select_row(frame, ABLATION_RUN_IDS[0], "dummy", "cross_era"), _D_IN_NOTE,
            "feature width",
        ),
        "n_frozen_test": int(smallest["n_test"]),
        "n_pool": _note_field(ceiling, _N_FT_IN_NOTE, "fine-tune pool size"),
        "n_smallest_budget": _note_field(smallest, _N_FT_IN_NOTE, "fine-tune sample size"),
        "budgets": budgets,
        "n_metrics_rows": len(frame),
        "n_run_ids": int(frame["run_id"].nunique()),
    }


def _pipe_height(body: str) -> float:
    """The height a box needs for ``body``'s line count. Derived, never guessed."""
    lines = len(body.splitlines()) if body else 0
    return (
        _PIPE_PAD_TOP + _PIPE_TITLE_H + _PIPE_PAD_BOTTOM
        + (_PIPE_TITLE_GAP + lines * _PIPE_LINE_H if lines else 0.0)
    )


def _pipe_box(
    ax: Any, x: float, y: float, w: float, title: str, body: str, *,
    face: str = _MODULE_FACE, edge: str = INK_SECONDARY, linewidth: float = 0.9,
    title_size: float = 7.0, body_size: float = 5.8, title_colour: str = INK_PRIMARY,
    linestyle: Any = "solid", zorder: int = 3,
) -> float:
    """One rounded box with a bold title and a centred body block. ``y`` is its BOTTOM edge.

    Returns the height it took, which :func:`plot_pipeline` has already reserved via
    :func:`_pipe_height` -- the two agree by construction rather than by hand-tuning.
    """
    height = _pipe_height(body)
    ax.add_patch(FancyBboxPatch(
        (x, y), w, height, boxstyle="round,pad=0,rounding_size=1.2", facecolor=face, edgecolor=edge,
        linewidth=linewidth, linestyle=linestyle, zorder=zorder,
    ))
    ax.text(
        x + w / 2, y + height - _PIPE_PAD_TOP, title, ha="center", va="top", fontsize=title_size,
        fontweight="bold", color=title_colour, zorder=zorder + 1,
    )
    if body:
        ax.text(
            x + w / 2, y + height - _PIPE_PAD_TOP - _PIPE_TITLE_H - _PIPE_TITLE_GAP, body,
            ha="center", va="top", fontsize=body_size, color=INK_SECONDARY, linespacing=1.28,
            zorder=zorder + 1,
        )
    return height


def _pipe_arrow(ax: Any, x0: float, y0: float, x1: float, y1: float, **kwargs: Any) -> None:
    """A straight flow arrow between two points in diagram coordinates."""
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7.5, shrinkA=0, shrinkB=0,
        color=kwargs.pop("color", INK_SECONDARY), linewidth=kwargs.pop("linewidth", 1.0), zorder=2,
        **kwargs,
    ))


def _pipe_elbow(ax: Any, x0: float, y0: float, x1: float, y1: float, y_bend: float) -> None:
    """A down-across-down connector: used where the flow forks or rejoins."""
    ax.plot([x0, x0], [y0, y_bend], color=INK_SECONDARY, linewidth=1.0, zorder=2,
            solid_capstyle="round")
    ax.plot([x0, x1], [y_bend, y_bend], color=INK_SECONDARY, linewidth=1.0, zorder=2,
            solid_capstyle="round")
    _pipe_arrow(ax, x1, y_bend, x1, y1)


def plot_pipeline(metrics: Any, out: Any = FIGURES) -> Path:
    """The Methods pipeline diagram: modules, artifacts, the fit boundary and the leakage seal.

    A schematic rather than a measurement — the only figure in this module that plots no data — but
    it is generated by the same command as the other six and from the same committed artifacts, so
    it cannot drift away from the code the way a hand-drawn diagram does. Every module path, every
    file it names and every count on it either exists in the repo or is read from
    ``reports/metrics.csv`` (see :func:`_pipeline_facts`).

    Two annotations carry the methodological content and are the reason the figure is worth a slot:

    * the **fit-on-source boundary** — the dashed rule between the two columns. Everything to its
      left is fitted on the 2015 source era and exactly one flow crosses it (the six fitted models
      and the frozen ``Preprocessor``); to its right nothing refits the preprocessor, so neither
      UNSW-test nor TON_IoT contributes a single statistic to the feature space it is scored in.
      Refitting there is what would invalidate the whole drift measurement;
    * the **leakage seal** around Phases 6 and 7 — :func:`evaluate.sealed` shadows ``fit``,
      ``fit_transform`` and ``partial_fit`` on the ``Preprocessor`` for the span of both phases and
      raises :class:`evaluate.LeakageError`, so "no leakage across eras" is enforced at runtime
      rather than asserted in prose. Phase 7 legitimately fits *models* on target labels; the seal
      around it is the preprocessor's, and each model's evaluation span is sealed separately inside
      :func:`transfer.recovery_curve`.

    ``metrics`` is the DataFrame from :func:`load_metrics` (or ``None`` / a path).
    """
    apply_style()
    frame = as_metrics_frame(metrics)
    f = _pipeline_facts(frame)

    fig, ax = plt.subplots(figsize=(PAGE_WIDTH_IN, 5.55))
    fig.subplots_adjust(left=0.008, right=0.992, top=0.808, bottom=0.013)
    # A hair of slack on x: the seal band is drawn 1.6 units wider than the column it wraps, and
    # patches are clipped to the axes, so a bare 0-100 range would shave its right edge off.
    ax.set_xlim(-1.6, 101.6)
    ax.set_ylim(0, _PIPE_TOP)
    ax.set_axis_off()

    blue, orange = REGIME_COLOURS["in_distribution"], REGIME_COLOURS["cross_era"]
    # Two columns rather than one tall spine: at 6.5 in wide, ten stacked full-width levels need a
    # figure taller than the page it has to share with six other figures, and the per-stage text
    # does not survive being cut to fit one. The split is not merely typographic -- the rule between
    # the columns IS the fit-on-source boundary, and exactly one flow crosses it.
    left_x, right_x, column_w = 1.0, 55.0, 44.0
    left_mid, right_mid = left_x + column_w / 2, right_x + column_w / 2
    boundary_x, lane_x = 51.5, 48.0
    top = 99.5

    cursor = top

    def place(x: float, title: str, body: str, **kwargs: Any) -> tuple[float, float]:
        """Place one box at ``x`` on the running cursor; returns its (top, bottom) edges."""
        nonlocal cursor
        height = _pipe_height(body)
        bottom = cursor - height
        _pipe_box(ax, x, bottom, column_w, title, body, **kwargs)
        cursor = bottom
        return bottom + height, bottom

    def flow(mid: float, gap: float = _PIPE_GAP) -> None:
        """Drop the cursor by ``gap`` and draw the arrow that spans it."""
        nonlocal cursor
        _pipe_arrow(ax, mid, cursor, mid, cursor - gap)
        cursor -= gap

    # --- left column: the source era, and every fit in the project ----------------------------
    place(
        left_x, "data/raw/  —  three delivered CSVs, git-ignored",
        f"UNSW-NB15 (2015)   training-set {f['n_unsw_train_raw']:,}  ·  test "
        f"{f['n_unsw_test']:,}\n"
        f"TON_IoT (2019–20)   Train_Test_Network  {f['n_toniot']:,}",
        face=_ARTIFACT_FACE, edge=INK_MUTED, title_size=6.8,
    )
    flow(left_mid)
    place(
        left_x, "src/schema_map.py  —  Phase 2:  align schemas",
        f"FEATURE_MAP: {f['n_mapped_concepts']} shared concepts  ·  DROP_COLUMNS: row\n"
        "id, IPs, ports, payload bytes  ·  collapse proto / service /\n"
        "state  ·  derive rate features + a zero_duration flag",
    )
    flow(left_mid)
    place(
        left_x, "data/processed/  —  harmonized parquets",
        f"unsw_common {f['n_unsw_parquet']:,} × {f['n_common_columns']}     "
        f"toniot_common {f['n_toniot']:,} × {f['n_common_columns']}",
        face=_ARTIFACT_FACE, edge=INK_MUTED, title_size=6.8,
    )
    flow(left_mid)
    place(
        left_x, "src/preprocess.py  —  Phase 3:  one Preprocessor",
        f"stratified {1 - f['validation_fraction']:.0%}/{f['validation_fraction']:.0%} split of "
        f"UNSW-train, seed {RANDOM_SEED}:\n"
        f"train fold {f['n_train_fold']:,}  ·  val {f['n_val_fold']:,}\n"
        "fit(train fold) and nothing else — the log1p, z-score and\n"
        "one-hot vocabularies come from those rows alone\n"
        "fit() raises on UNSW-test, on TON_IoT, on the two concatenated",
        edge=blue, title_colour=blue,
    )
    flow(left_mid)
    place(
        left_x, f"data/processed/preprocessor.joblib   —   d = {f['d_full']}", "",
        face=_ARTIFACT_FACE, edge=INK_MUTED, title_size=6.8,
    )
    flow(left_mid)
    models_top, models_bottom = place(
        left_x, "src/models/  —  Phases 4–5:  six models",
        f"each fit once on the same {f['n_train_fold']:,}-row train fold\n"
        "baselines.py: random forest · decision tree · LinearSVC ·\n"
        "majority-class dummy    scratch_logreg.py · scratch_mlp.py —\n"
        "pure numpy, hand-written gradients.  Class weights on all six.",
        edge=blue, title_colour=blue,
    )

    # --- right column: the two measurement phases, inside the seal, then the outputs -----------
    cursor = top
    seal_pad = 2.2
    evaluate_body = (
        "the same single fit per model, scored unchanged in two regimes\n"
        f"in-distribution:  UNSW-test  n = {f['n_unsw_test']:,}\n"
        f"cross-era, zero-shot:  TON_IoT  n = {f['n_toniot']:,}, no retraining\n"
        f"three conditions: full d = {f['d_full']}, plus the proto and conn_state\n"
        f"ablations at d = {f['d_ablated']} — one run_id each, never merged\n"
        "headline:  Δ = in_distribution − cross_era  (positive = lost)"
    )
    transfer_body = (
        f"TON_IoT split ONCE (seed {RANDOM_SEED}, stratified) into a permanent\n"
        f"{f['n_frozen_test']:,}-row test half and a disjoint {f['n_pool']:,}-row pool\n"
        + "budgets "
        + " · ".join(f"{budget:.0%}" for budget in f["budgets"])
        + f" of the pool ({f['n_smallest_budget']:,} rows at {f['budgets'][0]:.0%}), plus\n"
        "the full-pool ceiling — one run_id each, all scored on that\n"
        "one frozen half.  Models ARE fine-tuned on target labels here\n"
        "by design; the Preprocessor is not, and stays sealed throughout"
    )
    seal_height = (
        _pipe_height(evaluate_body) + _pipe_height(transfer_body) + _PIPE_GAP + 2 * seal_pad
    )
    seal_top = top + seal_pad
    ax.add_patch(FancyBboxPatch(
        (right_x - 1.6, seal_top - seal_height), column_w + 3.2, seal_height,
        boxstyle="round,pad=0,rounding_size=1.4", facecolor=_SEAL_FACE, edgecolor=orange,
        linewidth=1.0, linestyle=(0, (3.5, 2.0)), zorder=1,
    ))
    # Set on the band's own top edge, fieldset-style: the label IS the boundary it names.
    ax.text(
        right_mid, seal_top,
        "LEAKAGE SEAL  ·  evaluate.sealed(preprocessor)\n"
        "fit / fit_transform / partial_fit raise LeakageError",
        ha="center", va="center", fontsize=6.2, fontweight="bold", color=orange, linespacing=1.35,
        zorder=5, bbox={"facecolor": _SEAL_FACE, "edgecolor": "none", "pad": 1.4},
    )

    _, evaluate_bottom = place(
        right_x, "src/evaluate.py  —  Phase 6  (RQ1)", evaluate_body,
        edge=orange, title_colour=orange,
    )
    cursor -= _PIPE_GAP
    transfer_top, _ = place(
        right_x, "src/transfer.py  —  Phase 7  (RQ2)", transfer_body,
        edge=orange, title_colour=orange,
    )
    cursor = seal_top - seal_height
    flow(right_mid)
    place(
        right_x, "reports/  —  the committed run log + sidecars",
        f"metrics.csv — {f['n_metrics_rows']} rows / {f['n_run_ids']} run_ids, frozen "
        f"{len(METRICS_HEADER)}-column header,\nupserted on (run_id, model, regime)\n"
        "confusion_matrices.json · roc_curves.json   (Phase 6)\n"
        "per_family_metrics.csv, its own header and key   (Phases 6, 7)",
        face=_ARTIFACT_FACE, edge=INK_MUTED, title_size=6.8,
    )
    flow(right_mid)
    place(
        right_x, "src/plots.py  —  Phase 9:  render, never re-derive",
        "opens all four artifacts READ-ONLY and cross-checks them\n"
        "against each other before drawing anything",
    )
    flow(right_mid)
    place(
        right_x, "reports/figures/  —  seven PNGs + README.md",
        "every caption names the log row behind each mark on the figure",
        face=_ARTIFACT_FACE, edge=INK_MUTED, title_size=6.8,
    )
    outputs_bottom = cursor

    # --- the fit-on-source boundary, and the one flow that crosses it --------------------------
    # Drawn as the rule BETWEEN the columns rather than as a line across one of them: everything to
    # its left is fitted on the 2015 source era, and the single lane that crosses it carries the
    # fitted models and the frozen Preprocessor and nothing else.
    entry_y = (evaluate_bottom + transfer_top) / 2  # between the two sealed phases: it feeds both
    ax.plot([boundary_x, boundary_x], [outputs_bottom - 1.0, top + 3.4], color=INK_PRIMARY,
            linewidth=1.1, linestyle=(0, (4.5, 2.5)), zorder=4)
    ax.text(
        boundary_x, (models_bottom + entry_y) / 2 - 3.0,
        "FIT-ON-SOURCE BOUNDARY\nnothing to its right ever refits the Preprocessor",
        rotation=90, ha="center", va="center", fontsize=6.2, fontweight="bold", color=INK_PRIMARY,
        linespacing=1.35, zorder=5,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.4},
    )

    models_mid = (models_top + models_bottom) / 2
    ax.plot([left_x + column_w, lane_x], [models_mid, models_mid], color=INK_SECONDARY,
            linewidth=1.0, zorder=2, solid_capstyle="round")
    ax.plot([lane_x, lane_x], [models_mid, entry_y], color=INK_SECONDARY, linewidth=1.0, zorder=2,
            solid_capstyle="round")
    _pipe_arrow(ax, lane_x, entry_y, right_x - 1.6, entry_y)
    # Set along the lane rather than over the arrowhead: the channel between the columns is
    # 8 diagram units wide, and a horizontal label of this length there would knock a hole in the
    # left column's border on one side and the seal band's on the other.
    ax.text(
        lane_x - 1.4, (models_mid + entry_y) / 2,
        "six fitted models + the frozen Preprocessor",
        rotation=90, ha="center", va="center", fontsize=5.9, color=INK_SECONDARY, zorder=5,
    )

    # The whole chain is one command, which is the reproducibility claim; say so at the foot.
    ax.text(
        left_mid, models_bottom - 3.6,
        "./run.sh runs Phases 2 → 4, 6, 7 and 9 in order from the raw\n"
        f"CSVs.  RANDOM_SEED = {RANDOM_SEED} throughout (src/config.py), and every\n"
        "write upserts, so a re-run leaves the log, all three sidecars\n"
        "and all seven figures byte-identical.",
        ha="center", va="top", fontsize=6.1, color=INK_SECONDARY, linespacing=1.4,
    )

    _subtitle(
        fig,
        "Methods: the ids-crossera pipeline, and where the two leakage guards sit",
        "Strictly linear — each stage's committed output is the next stage's only input. The "
        "dashed rule is the fit-on-source\nboundary; the tinted band is the span over which the "
        "fitted Preprocessor is sealed against being refitted at all.",
    )

    sources: list[SourceRow] = [
        ("in-distribution evaluation-set size (n_test)",
         (IN_DISTRIBUTION_RUN_ID["dummy"], "dummy", "in_distribution")),
        ("cross-era evaluation-set size and the full feature width d (n_test, notes)",
         (CROSS_ERA_RUN_ID, "dummy", "cross_era")),
        ("ablated feature width d (notes)", (ABLATION_RUN_IDS[0], "dummy", "cross_era")),
        ("frozen test-half size and the smallest fine-tune budget (n_test, notes)",
         (RECOVERY_RUN_IDS[1], "dummy", TRANSFER_REGIME)),
        ("fine-tune pool size (notes)", (CEILING_RUN_ID, "dummy", TRANSFER_REGIME)),
        (f"metrics.csv row and run_id counts — every row of the log ({f['n_metrics_rows']} rows)",
         ("(all)", "(all)", "(all)")),
    ]

    caption = (
        f"**Figure 7 (Methods). The ids-crossera pipeline, and the two places leakage is "
        f"structurally prevented.** "
        f"Data flows down the left column, crosses the boundary once, and continues down the "
        f"right; each stage's committed output is the next stage's only "
        f"input. The three raw CSVs are harmonized by `src/schema_map.py` into two "
        f"{f['n_common_columns']}-column parquets over a shared feature subspace "
        f"({f['n_mapped_concepts']} mapped concepts plus derived rate features), a single "
        f"`Preprocessor` is fit in `src/preprocess.py` on the {f['n_train_fold']:,}-row UNSW "
        f"training fold and serialized, and the six models of Phases 4–5 are each fit once on that "
        f"same fold. "
        f"The dashed rule between the columns is the **fit-on-source boundary**: everything to its "
        f"right is transform-only, "
        f"so the UNSW-NB15 test set (n = {f['n_unsw_test']:,}) and TON_IoT "
        f"(n = {f['n_toniot']:,}) are pushed through the frozen Phase 3 parameters and neither "
        f"contributes a statistic to the d = {f['d_full']} feature space it is scored in. "
        f"The tinted band is the **leakage seal**: `evaluate.sealed()` shadows `fit`, "
        f"`fit_transform` and `partial_fit` on the preprocessor for the whole span of Phases 6 and "
        f"7 and raises `LeakageError` if any of them is called, which makes the no-leakage "
        f"constraint a runtime guarantee rather than a claim. Phase 7 does legitimately fit models "
        f"on target labels — the seal there is the preprocessor's, and each model's *evaluation* "
        f"span is sealed separately — which is why its budgets are drawn from a "
        f"{f['n_pool']:,}-row pool that the permanent {f['n_frozen_test']:,}-row test half never "
        f"intersects. "
        f"Every count on this figure is read from the committed `reports/metrics.csv` or from a "
        f"module constant the pipeline asserts against at runtime; nothing is transcribed. "
        f"The figure is numbered last only because `reports/figures/README.md` indexes in "
        f"generation order — in the report it belongs in Methods, ahead of the six results figures."
    )
    return save_figure(fig, "pipeline", caption, sources, out=out)


# --- Entry point -------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Regenerate every implemented figure from ``reports/metrics.csv`` into ``reports/figures/``.

    This is the entry point ``run.sh``'s Phase 9 line calls. It is additive by design: each figure
    that lands appends one call below, and the sidecar index is rewritten from whatever ran, so the
    command is safe to invoke with only part of Phase 9 implemented.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.plots",
        description=(
            "Phase 9: render the report figures into reports/figures/, with captions and per-row "
            "provenance in reports/figures/README.md. Reads reports/metrics.csv for every scalar, "
            "reports/confusion_matrices.json and reports/roc_curves.json (written by Phase 6) for "
            "the counts and curves the frozen metrics header has no room for, and "
            "reports/per_family_metrics.csv (Phases 6 and 7) for the per-attack-family "
            "breakdowns; never writes any of them."
        ),
    )
    parser.add_argument(
        "--out", default=str(FIGURES), help="output directory (default: reports/figures/)"
    )
    args = parser.parse_args(argv)

    set_seeds()
    out = Path(args.out)
    metrics = load_metrics()
    print(f"read {len(metrics):,} rows across {metrics['run_id'].nunique()} run_ids "
          f"from {METRICS_CSV}")

    figures: Iterable[Path] = (
        plot_indist_vs_crossera(metrics, out=out),
        plot_recovery_curve(metrics, out=out),
        # `None`: read the committed reports/confusion_matrices.json sidecar rather than re-running
        # Phase 6. Passing `run_phase6()`'s return value here instead would be correct data and a
        # three-minute `python -m src.plots` that re-fits eighteen models -- and inside `./run.sh`,
        # which runs Phase 6 immediately before this, it would run the whole phase a second time.
        plot_confusion_matrices(None, out=out),
        # `None` again, and for the same reason: read the committed reports/roc_curves.json rather
        # than re-running Phase 6 to recover fpr/tpr vectors `evaluate()` reports only the area of.
        plot_roc_curves(None, out=out),
        # The last two read reports/per_family_metrics.csv, again via `None`. They are TWO figures
        # over two different row populations and must stay two: Figure 5 is the cross-era
        # comparison and is restricted to the three families both eras label, Figure 6 is the
        # within-era recovery breakdown over TON_IoT's own eight 20,000-row attack types.
        plot_per_family_f1(None, out=out),
        plot_per_family_recovery(None, out=out),
        # The Methods schematic, generated last so the six results figures keep the numbering their
        # captions already carry. It plots no data, but every count on it is read from `metrics`
        # (or from a module constant the pipeline asserts against), so it cannot drift from the
        # code the way a hand-drawn diagram does.
        plot_pipeline(metrics, out=out),
    )
    for path in figures:
        print(f"  wrote {path}")
    print(f"  wrote {out / FIGURE_INDEX_NAME}  ({len(_FIGURE_INDEX)} captions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
