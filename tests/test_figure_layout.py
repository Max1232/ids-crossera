"""Phase 9's layout-override layer: the file's contract, and what it must not do to a figure.

``src/figure_layout.py`` exists to let a human fix a legend that sits on top of the data without
breaking the claim that ``./run.sh`` rebuilds every figure byte-identically — the position is
captured as committed data (``reports/figures/layout.json``) and re-applied on every render, rather
than baked into a hand-edited PNG. Four properties make that work, and each is a test below:

* **absence and emptiness are the coded defaults.** A clone with no ``layout.json`` must render
  exactly what ``src/plots.py`` says. If a missing file ever became an error, or silently shifted an
  artist, the committed figures and a fresh checkout would disagree.
* **a malformed file is loud.** The failure mode this guards is the quiet one: an override that is
  dropped for a typo'd key puts a figure back to a layout a human already rejected, and nothing in
  the output says so.
* **a saved position round-trips.** What the tuner measures after a drag has to be what the next
  render reproduces, or tuning is a one-way operation and the second ``./run.sh`` undoes it.
* **the writer is deterministic and additive.** ``layout.json`` is committed, so re-saving an
  unchanged tuning must leave it byte-identical, and tuning one figure must not discard another's.

Data-free (see ``tests/README.md``): every test builds its own throwaway figure with matplotlib's
``Agg`` backend and writes only under ``tmp_path``. Nothing here reads ``data/raw/`` or the committed
``reports/`` artifacts, and nothing writes to ``reports/``.
"""

from __future__ import annotations

import json

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from src.figure_layout import (  # noqa: E402
    LAYOUT_VERSION,
    Layout,
    _DraggableText,
    _moved,
    load_layout,
    save_layout,
)


@pytest.fixture
def figure():
    """A bare figure with one line, closed on teardown so a failing test cannot leak it."""
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ax.plot([0.0, 1.0], [0.0, 1.0], label="diagonal")
    yield fig, ax
    plt.close(fig)


# --- The file's contract -------------------------------------------------------------------------


def test_absent_file_means_coded_defaults(tmp_path):
    """No layout.json is the normal state of a fresh clone, not an error."""
    assert load_layout(tmp_path / "nope.json") == {}


def test_empty_file_means_coded_defaults(tmp_path):
    path = tmp_path / "layout.json"
    save_layout({}, path=path)
    assert load_layout(path) == {}


def test_a_wrong_version_is_rejected(tmp_path):
    path = tmp_path / "layout.json"
    path.write_text(json.dumps({"version": LAYOUT_VERSION + 1, "figures": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="version"):
        load_layout(path)


@pytest.mark.parametrize(
    "key, patch",
    [
        ("legend", {"position": [0.1, 0.1]}),   # a text's field on a legend
        ("text.floor", {"loc": [0.1, 0.1]}),    # a legend's field on a text
        ("legend", {"colour": "red"}),          # styling, which belongs in src/plots.py
    ],
)
def test_an_unsupported_key_is_rejected(tmp_path, key, patch):
    """Loudly, and naming the key: a silently ignored override reverts a tuned figure."""
    path = tmp_path / "layout.json"
    path.write_text(
        json.dumps({"version": LAYOUT_VERSION, "figures": {"fig": {key: patch}}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported key"):
        load_layout(path)


def test_a_note_field_is_allowed_and_never_reaches_matplotlib(tmp_path, figure):
    """Hand-editors need somewhere to say *why* a thing was moved; matplotlib must not see it."""
    fig, ax = figure
    path = tmp_path / "layout.json"
    path.write_text(
        json.dumps({
            "version": LAYOUT_VERSION,
            "figures": {"fig": {"legend": {"loc": [0.3, 0.4], "note": "clear of the curves"}}},
        }),
        encoding="utf-8",
    )
    layout = Layout.for_figure("fig", load_layout(path))
    layout.legend(ax, "legend", loc="upper right")  # would raise on an unexpected kwarg


def test_writes_are_deterministic_and_additive(tmp_path):
    """The file is committed: re-saving unchanged tuning must not churn the diff."""
    path = tmp_path / "layout.json"
    save_layout({"one": {"legend": {"loc": [0.25, 0.5]}}}, path=path)
    first = path.read_bytes()
    save_layout({"one": {"legend": {"loc": [0.25, 0.5]}}}, path=path)
    assert path.read_bytes() == first

    # Tuning a second figure must not discard the first's.
    save_layout({"two": {"legend": {"loc": [0.1, 0.1]}}}, path=path)
    assert set(load_layout(path)) == {"one", "two"}
    assert load_layout(path)["one"]["legend"]["loc"] == [0.25, 0.5]


def test_a_subpixel_difference_is_rounded_away(tmp_path):
    path = tmp_path / "layout.json"
    save_layout({"one": {"legend": {"loc": [0.1234567891, 0.5]}}}, path=path)
    assert load_layout(path)["one"]["legend"]["loc"] == [0.123457, 0.5]


# --- Application ---------------------------------------------------------------------------------


def test_a_legend_override_moves_the_legend(figure):
    fig, ax = figure
    default = Layout.for_figure("fig", {})
    default.legend(ax, "legend", loc="upper left")
    before = default.measure(fig)["legend"]["loc"]

    moved = Layout.for_figure("fig", {"fig": {"legend": {"loc": [0.42, 0.17]}}})
    moved.legend(ax, "legend", loc="upper left")
    after = moved.measure(fig)["legend"]["loc"]

    assert before != pytest.approx(after)
    assert after == pytest.approx([0.42, 0.17], abs=1e-3)


def test_a_point_loc_drops_the_call_sites_anchor(figure):
    """`loc=(x, y)` composes with `bbox_to_anchor`, so a dragged position must clear the anchor.

    Without this the saved coordinates would be offset by whatever anchor the figure function
    happened to pass -- for the figure legends in `src/plots.py` that is (0.5, 0.004), i.e. half a
    figure width off.
    """
    fig, ax = figure
    layout = Layout.for_figure("fig", {"fig": {"legend": {"loc": [0.42, 0.17]}}})
    layout.legend(ax, "legend", loc="lower center", bbox_to_anchor=(0.5, 0.004))
    assert layout.measure(fig)["legend"]["loc"] == pytest.approx([0.42, 0.17], abs=1e-3)


def test_a_text_override_moves_the_text(figure):
    fig, ax = figure
    layout = Layout.for_figure("fig", {"fig": {"text.floor": {"position": [0.8, 0.2]}}})
    artist = layout.text(ax, "text.floor", 0.1, 0.9, "floor 0.4444")
    assert artist.get_position() == (0.8, 0.2)
    assert layout.measure(fig)["text.floor"]["position"] == pytest.approx([0.8, 0.2])


def test_a_legend_key_must_be_named_as_one(figure):
    """The `legend` prefix is what `load_layout` validates a patch's schema against."""
    _, ax = figure
    layout = Layout.for_figure("fig", {})
    with pytest.raises(ValueError, match="must start with 'legend'"):
        layout.legend(ax, "roc_legend", loc="upper left")


def test_a_figure_legend_is_measured_in_figure_fractions(figure):
    """Two frames are in play and confusing them silently halves or doubles every coordinate."""
    fig, ax = figure
    layout = Layout.for_figure("fig", {})
    layout.legend(fig, "legend.shared", loc="lower center", bbox_to_anchor=(0.5, 0.004))
    x, y = layout.measure(fig)["legend.shared"]["loc"]
    assert 0.0 < x < 1.0 and 0.0 <= y < 0.2  # near the bottom centre *of the figure*


# --- Round-trip ----------------------------------------------------------------------------------


def test_a_measured_position_round_trips(figure):
    """Re-applying what the tuner measured must reproduce the same position.

    This is the property the whole mechanism rests on: tune once, and every later `./run.sh`
    reproduces the tuned layout rather than drifting back toward the coded default.
    """
    fig, ax = figure
    first = Layout.for_figure("fig", {})
    first.legend(ax, "legend", loc="upper left", bbox_to_anchor=(-0.005, -0.135))
    first.text(ax, "text.note", 0.3, 0.7, "note")
    measured = first.measure(fig)

    fig2, ax2 = plt.subplots(figsize=(4.0, 3.0))
    try:
        ax2.plot([0.0, 1.0], [0.0, 1.0], label="diagonal")
        second = Layout.for_figure("fig", {"fig": measured})
        second.legend(ax2, "legend", loc="upper left", bbox_to_anchor=(-0.005, -0.135))
        second.text(ax2, "text.note", 0.3, 0.7, "note")
        again = second.measure(fig2)
    finally:
        plt.close(fig2)

    for key in measured:
        field = "loc" if "loc" in measured[key] else "position"
        assert again[key][field] == pytest.approx(measured[key][field], abs=1e-6), key


def test_moved_is_the_tolerance_the_tuner_saves_on():
    """An unmoved artist must not be written back -- see the note in `figure_layout.tune`."""
    assert not _moved({"loc": [0.5, 0.5]}, {"loc": [0.5, 0.50000001]})
    assert _moved({"loc": [0.5, 0.5]}, {"loc": [0.5, 0.52]})
    assert _moved(None, {"loc": [0.5, 0.5]})


def test_dragging_a_text_writes_back_through_its_own_transform(figure):
    """The drag arithmetic, exercised on synthetic events rather than a GUI.

    A text placed in *data* coordinates must come back in data coordinates after a drag measured in
    pixels; getting this wrong would persist screen pixels into a file the next render reads as data
    units, which is a position that moves whenever the figure is resized.
    """
    from matplotlib.backend_bases import MouseEvent, PickEvent

    fig, ax = figure
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    artist = ax.text(0.2, 0.2, "floor")
    fig.canvas.draw()

    handle = _DraggableText(artist)
    start = ax.transData.transform((0.2, 0.2))
    target = ax.transData.transform((0.6, 0.5))
    press = MouseEvent("button_press_event", fig.canvas, *start, button=1)
    handle.on_pick(PickEvent("pick_event", fig.canvas, press, artist))
    handle.on_motion(MouseEvent("motion_notify_event", fig.canvas, *target, button=1))
    handle.on_release(MouseEvent("button_release_event", fig.canvas, *target, button=1))

    # To within a pixel, which is the resolution a drag has: matplotlib rounds every mouse event to
    # integer device coordinates, so the round-trip through `transData` cannot be exact.
    pixel = ax.transData.inverted().transform((1.0, 1.0)) - ax.transData.inverted().transform((0, 0))
    assert artist.get_position() == pytest.approx((0.6, 0.5), abs=max(pixel))


# --- The output modes ----------------------------------------------------------------------------
#
# These render real figures, so unlike everything above they read the committed `reports/` artifacts
# (never `data/raw/`, and never writing outside `tmp_path`). They are about *what lands on disk* in
# each of the three modes -- the drawing itself is covered by the figures being committed.


@pytest.fixture
def rendered(tmp_path):
    """Render one figure into ``tmp_path`` and return a re-render callable plus its path."""
    from src import plots

    def render(*argv: str) -> int:
        return plots.main([*argv, "--out", str(tmp_path), "roc_curves"])

    return render, tmp_path / "roc_curves.png"


def test_rendering_twice_is_byte_identical(rendered):
    """The claim `./run.sh` is checked against, at the granularity of one figure."""
    render, path = rendered
    assert render() == 0
    first = path.read_bytes()
    assert render() == 0
    assert path.read_bytes() == first


def test_vector_export_does_not_disturb_the_committed_png(rendered):
    """`--vector` is additive: the PNG the report embeds must be the same file either way."""
    from src.plots import EDITABLE_SUBDIR, VECTOR_FORMATS

    render, path = rendered
    assert render() == 0
    png = path.read_bytes()
    assert render("--vector") == 0
    assert path.read_bytes() == png
    for fmt in VECTOR_FORMATS:
        assert (path.parent / EDITABLE_SUBDIR / f"roc_curves.{fmt}").exists()


def test_a_partial_render_leaves_the_caption_sidecar_alone(rendered):
    """Naming one figure must not truncate README.md to that figure's caption -- it is committed."""
    from src.plots import FIGURE_INDEX_NAME

    render, path = rendered
    assert render() == 0
    assert not (path.parent / FIGURE_INDEX_NAME).exists()


def test_an_unknown_figure_name_is_rejected(tmp_path):
    from src import plots

    with pytest.raises(SystemExit, match="unknown figure name"):
        plots.main(["--out", str(tmp_path), "roc_curvez"])


def test_the_registry_names_the_seven_committed_figures():
    import pandas as pd

    from src.evaluate import METRICS_HEADER
    from src.plots import builders

    names = list(builders(pd.DataFrame(columns=list(METRICS_HEADER))))
    assert len(names) == 7
    assert names[-1] == "pipeline"  # generated last so the results figures keep their numbering


@pytest.mark.parametrize(
    "name, expected",
    [
        ("roc_curves", {"legend.in_distribution", "legend.cross_era", "text.chance_note"}),
        ("drift_indist_vs_crossera",
         {"legend", "text.chance", "text.dummy_f1.in_f1", "text.dummy_f1.cross_f1"}),
        ("recovery_curve", {"legend", "text.floor.roc_auc", "text.floor.f1"}),
    ],
)
def test_each_figure_registers_its_movable_artists(tmp_path, name, expected):
    """Every legend and reference-line label goes through `Layout`, not straight to matplotlib.

    A figure edited to call `ax.legend` directly would still render, and would silently stop being
    tunable -- which is the regression this pins.
    """
    from src import plots

    previous = plots.set_session(plots.RenderSession(formats=(), keep_open=True))
    try:
        plots.builders(plots.load_metrics())[name](tmp_path)
        _, figure, layout = plots.session().held[-1]
        assert expected <= {placement.key for placement in layout.placements}
    finally:
        plt.close("all")
        plots.set_session(previous)
