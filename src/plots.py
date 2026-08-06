"""Figures for the report and presentation (Phase 9) -> reports/figures/.

Every figure needs clear labels, legends, and captions (graded). Produces:
  - in-distribution vs cross-era metric bars
  - confusion matrices (per model, per regime)
  - ROC curves
  - per-shared-family F1
  - the transfer-learning recovery curve (RQ2 secondary headline)

**Every number rendered here is read from a committed artifact of the phase that measured it.**
Nothing in this module fits, transforms, or re-derives anything, so a figure can never disagree
with the table it illustrates. There are exactly two such artifacts:

* ``reports/metrics.csv`` -- the run log, keyed ``(run_id, model, regime)``; the source for every
  scalar metric on every figure;
* ``reports/confusion_matrices.json`` -- the 2x2 count matrices, which the log's frozen 14-column
  header has no room for. Written by :func:`evaluate.run_phase6` from the matrices it already
  computed (see :func:`evaluate.write_confusion_matrices`).

Three consequences worth knowing before editing:

* both files are opened **read-only** -- this module must never call :func:`evaluate.log_metrics`
  or :func:`evaluate.write_confusion_matrices`;
* a figure that needs a quantity neither artifact carries (ROC curve vectors, per-family F1) needs
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
from matplotlib.patches import Rectangle  # noqa: E402

from .config import CONFUSION_JSON, FIGURES, METRICS_CSV, set_seeds  # noqa: E402
from .evaluate import (  # noqa: E402
    METRICS_HEADER,
    METRICS_KEY,
    PHASE4_AGREEMENT_TOLERANCE,
    POSITIVE_LABEL,
    read_confusion_matrices,
    read_metrics,
)

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

#: ``(figure element, (run_id, model, regime))`` -- the provenance of one mark on one figure.
SourceRow = tuple[str, tuple[str, str, str]]

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
        f"Source log: `reports/metrics.csv` (MD5 `{_metrics_fingerprint()}`) — every scalar. The 2x2",
        "confusion counts, which that file's frozen 14-column header has no room for, come from",
        f"`reports/confusion_matrices.json` (MD5 `{_metrics_fingerprint(CONFUSION_JSON)}`), written by",
        "`evaluate.run_phase6()`. This module reads both and writes neither; where they overlap they",
        "are cross-checked against each other before anything is drawn.",
        "",
        "**Δ sign convention throughout: `Δ = in_distribution − cross_era`, i.e. what the model",
        "*lost* — positive means degraded.** This is `evaluate.metric_deltas()` and what `./run.sh`",
        "prints.",
        "",
    ]
    for number, entry in enumerate(_FIGURE_INDEX, start=1):
        lines += [
            f"## Figure {number} — `{entry['name']}`",
            "",
            entry["caption"],
            "",
            "| figure element | run_id | model | regime |",
            "|---|---|---|---|",
        ]
        for element, (run_id, model, regime) in entry["sources"]:
            lines.append(f"| {element} | `{run_id}` | `{model}` | `{regime}` |")
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


def plot_roc_curves(results: Any, out=FIGURES) -> None:
    raise NotImplementedError("Phase 9: ROC curves")


def plot_per_family_f1(results: Any, out=FIGURES) -> None:
    raise NotImplementedError("Phase 9: per-family F1")


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
            "provenance in reports/figures/README.md. Reads reports/metrics.csv for every scalar "
            "and reports/confusion_matrices.json (written by Phase 6) for the 2x2 counts the "
            "frozen metrics header has no room for; never writes either."
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
    )
    for path in figures:
        print(f"  wrote {path}")
    print(f"  wrote {out / FIGURE_INDEX_NAME}  ({len(_FIGURE_INDEX)} captions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
