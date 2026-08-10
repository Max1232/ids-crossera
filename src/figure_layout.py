"""Human-editable placement for the Phase 9 figures: committed overrides + an interactive tuner.

Matplotlib places a legend where you tell it to, and at half-page width "where the curves are not"
is a judgement a human eye makes better than a hardcoded ``loc=``. But this project's load-bearing
claim is that ``./run.sh`` rebuilds every figure **byte-identically** from the raw CSVs, so a
hand-edited PNG cannot be the artifact the report embeds -- the moment a human moves something in
an image editor, the pipeline no longer reproduces what a grader sees.

This module is the way out of that trade-off. It splits "where does this artist go" out of
:mod:`src.plots` and into a committed data file, ``reports/figures/layout.json``:

1. every legend and every reference-line label that has ever collided with data is placed through
   :class:`Layout` rather than by a literal ``loc=`` at the call site;
2. ``python -m src.plots --tune`` opens the figures in a real window where those artists are
   **draggable**, and pressing ``s`` writes the positions you dragged them to into that JSON;
3. every later run -- including ``./run.sh`` -- reads the JSON back and renders the human-tuned
   layout deterministically.

So the human judgement is captured once, as data, and the reproducibility claim survives: the
figure in the report is still pipeline output. ``layout.json`` is committed for exactly the same
reason the figures are.

An override is a **patch**, never a replacement: the call site in :mod:`src.plots` keeps its full
default styling, and the JSON supplies only the handful of keys a human moved. An empty or absent
``layout.json`` therefore renders precisely the coded defaults, which is what makes the file safe to
delete and safe to hand-edit.

Two coordinate conventions, both chosen so a value round-trips through the JSON unchanged:

* **legends** persist ``{"loc": [x, y]}`` -- the lower-left corner of the legend box as a fraction
  of its parent (the axes for an ``ax.legend``, the whole figure for a ``fig.legend``). Supplying a
  2-element ``loc`` also *drops* any ``bbox_to_anchor`` the call site passed, because the two
  compose and leaving both in place would offset the dragged position by the anchor point;
* **texts** persist ``{"position": [x, y]}`` in whatever transform the text was created with --
  data coordinates for the usual ``ax.text``, axes fractions for one created with
  ``transform=ax.transAxes``. The tuner reads and writes through that same transform, so it never
  has to know which it was.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from matplotlib.figure import Figure
from matplotlib.offsetbox import DraggableBase

from .config import FIGURES

#: The committed override file. Written only by ``python -m src.plots --tune`` (and by hand, which
#: is a supported way to use it); read by every render.
LAYOUT_JSON: Path = FIGURES / "layout.json"

#: Bumped if the on-disk shape ever changes. :func:`load_layout` refuses a version it does not know
#: rather than silently ignoring overrides -- a figure quietly reverting to its coded default is
#: exactly the kind of drift this file exists to prevent.
LAYOUT_VERSION: int = 1

#: Keys a legend override may carry. Deliberately short: this file tunes *placement*, and letting
#: it set arbitrary kwargs would move styling decisions out of the code that documents them.
LEGEND_KEYS: frozenset[str] = frozenset({"loc", "bbox_to_anchor", "ncol", "fontsize"})

#: Keys a text override may carry.
TEXT_KEYS: frozenset[str] = frozenset({"position", "ha", "va", "fontsize"})


# --- The file ------------------------------------------------------------------------------------


def load_layout(path: Any = LAYOUT_JSON) -> dict[str, dict[str, dict[str, Any]]]:
    """Read ``layout.json`` into ``{figure_name: {element_key: patch}}``.

    A missing file is not an error -- it means "render the coded defaults", which is the state a
    fresh clone is in until someone tunes something. A malformed one *is* an error: silently
    dropping a tuned position would put the committed figures back to a layout a human already
    rejected, with nothing in the output to say so.
    """
    path = Path(path)
    if not path.exists():
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    version = document.get("version")
    if version != LAYOUT_VERSION:
        raise ValueError(
            f"{path} has version {version!r}, expected {LAYOUT_VERSION}. Re-tune it with "
            f"`python -m src.plots --tune` rather than editing the version by hand."
        )
    figures = document.get("figures", {})
    if not isinstance(figures, dict):
        raise ValueError(f"{path}: 'figures' must be an object, got {type(figures).__name__}")
    for name, elements in figures.items():
        if not isinstance(elements, dict):
            raise ValueError(f"{path}: figures.{name} must be an object")
        for key, patch in elements.items():
            if not isinstance(patch, dict):
                raise ValueError(f"{path}: figures.{name}.{key} must be an object")
            allowed = LEGEND_KEYS if key.startswith("legend") else TEXT_KEYS
            unknown = set(patch) - allowed - {"note"}
            if unknown:
                raise ValueError(
                    f"{path}: figures.{name}.{key} has unsupported key(s) {sorted(unknown)}; "
                    f"allowed: {sorted(allowed)}"
                )
    return figures


def save_layout(figures: dict[str, dict[str, Any]], path: Any = LAYOUT_JSON) -> Path:
    """Write ``figures`` to ``layout.json``, merging into whatever is already there.

    Merged rather than replaced so tuning one figure cannot discard another's tuning -- the tuner is
    usually pointed at the single figure that needs work.

    Deterministic on purpose: sorted keys, fixed indent, six decimal places, one trailing newline.
    The file is committed alongside the figures it positions, so an unchanged tuning session must
    leave it byte-identical.
    """
    path = Path(path)
    merged = load_layout(path)
    for name, elements in figures.items():
        merged.setdefault(name, {}).update(elements)
    document = {
        "version": LAYOUT_VERSION,
        "_comment": (
            "Human-tuned artist placement for reports/figures/, read by src/plots.py on every "
            "render so ./run.sh reproduces the tuned layout byte-identically. Regenerate with "
            "`python -m src.plots --tune [figure ...]`, drag, then press s. Hand-editing is fine: "
            "legend 'loc' is [x, y] of the box's lower-left as a fraction of its parent; text "
            "'position' is [x, y] in that text's own transform. Delete an entry to fall back to "
            "the coded default in src/plots.py."
        ),
        "figures": _rounded(merged),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _rounded(value: Any, decimals: int = 6) -> Any:
    """Round every float in a nested structure, so a sub-pixel drag cannot churn the diff."""
    if isinstance(value, float):
        return round(value, decimals)
    if isinstance(value, dict):
        return {k: _rounded(v, decimals) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded(v, decimals) for v in value]
    return value


# --- Placement -----------------------------------------------------------------------------------


@dataclass
class Placement:
    """One artist :class:`Layout` placed, kept so the tuner can make it draggable and read it back.

    ``frame`` is the transform a legend's persisted ``loc`` is a fraction of; it is unused for
    texts, which carry their own transform.
    """

    kind: str  # "legend" | "text"
    key: str
    artist: Any
    frame: Any = None
    overridden: bool = False


@dataclass
class Layout:
    """Placement helper for one figure: applies committed overrides, records what it placed.

    Construct one per figure function, keyed on the figure's file stem (``"roc_curves"``), and route
    every legend and collision-prone label through :meth:`legend` and :meth:`text` instead of
    calling matplotlib directly. The call site still owns all the styling -- this only lets a
    committed override move things.
    """

    figure: str
    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    placements: list[Placement] = field(default_factory=list)

    @classmethod
    def for_figure(cls, name: str, layout: dict[str, dict[str, Any]] | None = None) -> "Layout":
        """Build the :class:`Layout` for one figure, reading ``layout.json`` unless given a dict."""
        table = load_layout() if layout is None else layout
        return cls(figure=name, overrides=dict(table.get(name, {})))

    # -- application ----------------------------------------------------------------------------

    def patch(self, key: str) -> dict[str, Any]:
        """The committed patch for ``key``, minus the human-only ``note`` field."""
        return {k: v for k, v in self.overrides.get(key, {}).items() if k != "note"}

    def legend(
        self,
        target: Any,
        key: str,
        handles: Sequence[Any] | None = None,
        labels: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Draw a legend on ``target`` (an ``Axes`` or a ``Figure``), applying any override.

        ``key`` must start with ``"legend"`` -- :func:`load_layout` uses that prefix to decide which
        keys a patch may carry, so a mis-named key would validate against the wrong schema.
        """
        if not key.startswith("legend"):
            raise ValueError(f"legend key must start with 'legend', got {key!r}")
        patch = self.patch(key)
        kwargs.update(patch)
        # A 2-element `loc` is a point in the parent frame and composes with `bbox_to_anchor`, so a
        # dragged position would be offset by whatever anchor the call site passed. The tuner writes
        # positions in the parent frame precisely so that dropping the anchor is the correct move.
        if _is_point(patch.get("loc")):
            kwargs.pop("bbox_to_anchor", None)
            kwargs["loc"] = tuple(patch["loc"])
        args = () if handles is None else (handles, labels)
        legend = target.legend(*args, **kwargs)
        # The frame a persisted `loc` is a fraction of: the whole figure for a `fig.legend`, the
        # panel for an `ax.legend`. Confusing the two silently rescales every saved coordinate.
        frame = target.transFigure if isinstance(target, Figure) else target.transAxes
        self.placements.append(
            Placement("legend", key, legend, frame=frame, overridden=bool(patch))
        )
        return legend

    def text(self, target: Any, key: str, x: float, y: float, s: str, **kwargs: Any) -> Any:
        """Draw a text on ``target``, applying any override to its position and alignment."""
        patch = self.patch(key)
        position = patch.pop("position", None)
        if position is not None:
            x, y = float(position[0]), float(position[1])
        kwargs.update(patch)
        artist = target.text(x, y, s, **kwargs)
        overridden = bool(patch) or position is not None
        self.placements.append(Placement("text", key, artist, overridden=overridden))
        return artist

    # -- read-back ------------------------------------------------------------------------------

    def measure(self, figure: Any) -> dict[str, dict[str, Any]]:
        """Where everything recorded actually ended up, in the form :func:`save_layout` persists.

        Requires a drawn figure: a legend's box is only known once a renderer has laid it out.
        """
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        measured: dict[str, dict[str, Any]] = {}
        for placement in self.placements:
            if placement.kind == "legend":
                box = placement.artist.get_window_extent(renderer)
                x, y = placement.frame.inverted().transform((box.x0, box.y0))
                measured[placement.key] = {"loc": [float(x), float(y)]}
            else:
                x, y = placement.artist.get_position()
                measured[placement.key] = {"position": [float(x), float(y)]}
        return measured


def _moved(before: dict[str, Any] | None, after: dict[str, Any], atol: float = 1e-4) -> bool:
    """True if two measured patches differ by more than a mouse could plausibly not have moved.

    The tolerance is in the persisted units -- axes or figure fractions for a legend, data units for
    a text -- so it is deliberately coarse: a drag of less than 1e-4 of a panel is not a drag.
    """
    if before is None:
        return True
    field_name = "loc" if "loc" in after else "position"
    old, new = before.get(field_name), after.get(field_name)
    if old is None or new is None:
        return True
    return any(abs(a - b) > atol for a, b in zip(old, new))


def _is_point(value: Any) -> bool:
    """True for a 2-element sequence of numbers -- a ``loc`` given as coordinates, not a name."""
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(v, (int, float)) for v in value)
    )


# --- The tuner -----------------------------------------------------------------------------------

#: Printed once when a tuning window opens. Kept here rather than in `plots` so the key bindings and
#: the handlers that implement them cannot drift apart.
TUNE_HELP: str = f"""\
  drag   move any legend or dashed-line label
  s      save what you moved to {LAYOUT_JSON.name} (artists you did not move are left alone)
  r      put everything back where this window opened
  q      close without saving
"""


class _DraggableText(DraggableBase):
    """Drag a :class:`~matplotlib.text.Text`, writing the result back through its own transform.

    The transform round-trip is the whole point: these labels are positioned in *data* coordinates,
    and a drag arrives in device pixels. Persisting the pixels would give a position that moves
    whenever the figure is resized.
    """

    def save_offset(self) -> None:
        self._origin = self.ref_artist.get_transform().transform(self.ref_artist.get_position())

    def update_offset(self, dx: float, dy: float) -> None:
        target = (self._origin[0] + dx, self._origin[1] + dy)
        self.ref_artist.set_position(
            tuple(self.ref_artist.get_transform().inverted().transform(target))
        )


def make_draggable(figure: Any, layout: Layout) -> list[Any]:
    """Make every artist ``layout`` placed draggable in an interactive window.

    Legends use matplotlib's own :meth:`Legend.set_draggable`; texts use :class:`_DraggableText`,
    which is the same :class:`~matplotlib.offsetbox.DraggableBase` machinery -- so both go through
    one pick/motion/release implementation rather than a hand-rolled second one.
    """
    handles = []
    for placement in layout.placements:
        if placement.kind == "legend":
            placement.artist.set_draggable(True, use_blit=False)
            handles.append(placement.artist)
        else:
            handles.append(_DraggableText(placement.artist))
    return handles


def restore(layout: Layout, positions: dict[str, dict[str, Any]]) -> None:
    """Put every artist back to ``positions``, as returned by :meth:`Layout.measure`.

    Used by the tuner's reset key. Writes through the same two coordinate conventions
    :meth:`Layout.measure` reads, so a measure/restore pair is a no-op.
    """
    for placement in layout.placements:
        patch = positions.get(placement.key)
        if patch is None:  # pragma: no cover - `measure` covers every placement
            continue
        if placement.kind == "legend":
            placement.artist.set_loc(tuple(patch["loc"]))
        else:
            placement.artist.set_position(tuple(patch["position"]))


def tune(
    figure: Any,
    layout: Layout,
    path: Any = LAYOUT_JSON,
    show: bool = True,
) -> list[Any]:
    """Open one figure for interactive tuning; ``s`` saves, ``r`` resets, ``q`` closes.

    Returns the drag handles (the caller must keep them alive -- matplotlib's draggable machinery
    holds only a weak claim on them through the canvas callback registry).
    """
    import matplotlib.pyplot as plt

    handles = make_draggable(figure, layout)
    # Where everything sits before the human touches it, so a save can persist only what moved.
    # Writing back an unmoved default would look harmless and is not: `loc=(x, y)` and the call
    # site's `loc=... bbox_to_anchor=...` are different code paths through matplotlib's layout, and
    # they land in the same place to within a rounding error -- enough to re-anti-alias the PNG for
    # no visible change. An entry already in the committed file is kept whether or not it moved,
    # since dropping it would silently revert that tuning.
    baseline = layout.measure(figure)
    already = {p.key for p in layout.placements if p.overridden}
    print(f"  tuning {layout.figure}  ({len(layout.placements)} movable artists)")
    print(TUNE_HELP, end="")

    def on_key(event: Any) -> None:
        if event.key == "s":
            measured = {
                key: patch
                for key, patch in layout.measure(figure).items()
                if key in already or _moved(baseline.get(key), patch)
            }
            if not measured:
                print("  nothing moved — layout.json left alone")
                return
            written = save_layout({layout.figure: measured}, path=path)
            print(f"  saved {len(measured)} position(s) for {layout.figure} -> {written}")
            for key, patch in sorted(measured.items()):
                coords = patch.get("loc") or patch.get("position")
                print(f"    {key:<34} {[round(c, 4) for c in coords]}")
            print("  re-render with `python -m src.plots` to write the tuned PNG")
        elif event.key == "r":
            restore(layout, baseline)
            figure.canvas.draw_idle()
            print("  reset to the layout this window opened with (nothing saved)")
        elif event.key == "q":
            print("  closed without saving")
            plt.close(figure)

    figure.canvas.mpl_connect("key_press_event", on_key)
    if show:
        plt.show()
    return handles


def interactive_backend(candidates: Iterable[str] = ("TkAgg", "QtAgg", "GTK4Agg")) -> str:
    """Switch matplotlib to the first GUI backend that imports, and return its name.

    :mod:`src.plots` selects ``Agg`` at import time -- ``run.sh`` has no display and figures are
    files on disk by design -- so tuning has to force a real backend. Forcing it here, after import
    but before any figure exists, is the supported way round that.
    """
    import matplotlib

    for name in candidates:
        try:
            matplotlib.use(name, force=True)
            return name
        except Exception:  # pragma: no cover - depends on what GUI toolkits are installed
            continue
    raise RuntimeError(  # pragma: no cover - tkinter ships with CPython on every target platform
        "no interactive matplotlib backend available (tried "
        f"{', '.join(candidates)}). On Debian/Ubuntu: `sudo apt install python3-tk`. Without a "
        "display, use `python -m src.plots --vector` and edit the SVGs instead."
    )
