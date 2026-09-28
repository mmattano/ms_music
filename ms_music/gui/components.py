"""Reusable NiceGUI pieces: a local file picker, a dynamic parameter form,
and helpers that turn engine output into things the browser can show.

Everything here that touches Matplotlib or numpy (``render_figure_png``,
``waveform_svg``) is plain Python so it can run on a worker thread;
only the ``ui.*`` builders must run on the event loop.
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

from nicegui import ui

from .controller import parse_value


# ------------------------------------------------------------ file picker
class LocalFilePicker(ui.dialog):
    """Browse the local filesystem and pick a file.

    The app only runs locally, so we can list directories server-side and
    hand the engine a real path. That avoids uploading multi-GB mzML files
    through the browser and works the same in browser and native mode.
    Usage: ``path = await LocalFilePicker(start_dir, extensions)``.
    """

    def __init__(self, start_dir=None, extensions=None):
        super().__init__()
        self.extensions = {e.lower() for e in (extensions or [])}
        start = Path(start_dir).expanduser() if start_dir else Path.home()
        self.path = start if start.is_dir() else start.parent
        if not self.path.is_dir():
            self.path = Path.home()
        self.show_hidden = False

        with self, ui.card().classes("w-[40rem] max-w-full"):
            with ui.row().classes("w-full items-center no-wrap"):
                ui.button(icon="arrow_upward", on_click=self._go_up).props(
                    "flat round dense"
                ).tooltip("Parent folder")
                ui.button(
                    icon="home", on_click=lambda: self._go(Path.home())
                ).props("flat round dense").tooltip("Home")
                self.path_label = ui.label().classes(
                    "text-sm font-mono truncate grow"
                )
            self.listing = (
                ui.list()
                .props("dense separator")
                .classes("w-full h-96 overflow-auto")
            )
            with ui.row().classes("w-full items-center justify-between"):
                ui.checkbox("Show hidden", on_change=self._toggle_hidden)
                if self.extensions:
                    ui.label(
                        "Showing: " + ", ".join(sorted(self.extensions))
                    ).classes("text-xs text-grey")
                ui.button("Cancel", on_click=lambda: self.submit(None)).props(
                    "flat"
                )
        self._refresh()

    def _toggle_hidden(self, e):
        self.show_hidden = e.value
        self._refresh()

    def _go_up(self):
        self._go(self.path.parent)

    def _go(self, path):
        self.path = path
        self._refresh()

    def _visible(self, entry: os.DirEntry) -> bool:
        if not self.show_hidden and entry.name.startswith("."):
            return False
        if entry.is_dir():
            return True
        if not self.extensions:
            return True
        return os.path.splitext(entry.name)[1].lower() in self.extensions

    def _refresh(self):
        self.path_label.text = str(self.path)
        self.listing.clear()
        try:
            entries = [e for e in os.scandir(self.path) if self._visible(e)]
        except OSError as exc:
            with self.listing:
                ui.item(f"Cannot open folder: {exc}").classes("text-negative")
            return
        entries.sort(key=lambda e: (not e.is_dir(), e.name.lower()))
        with self.listing:
            if not entries:
                ui.item("(empty)").classes("text-grey")
            for entry in entries:
                is_dir = entry.is_dir()
                target = Path(entry.path)
                handler = (
                    (lambda p=target: self._go(p))
                    if is_dir
                    else (lambda p=target: self.submit(str(p)))
                )
                with ui.item(on_click=handler):
                    with ui.item_section().props("avatar"):
                        ui.icon(
                            "folder" if is_dir else "description",
                            color="primary" if is_dir else None,
                        )
                    ui.item_section(entry.name)


# ------------------------------------------------------------- param form
class ParamForm(ui.column):
    """Builds inputs from a ``[(name, default), ...]`` spec.

    - booleans become switches, ints/floats number fields;
    - names listed in ``choices`` become dropdowns;
    - strings are plain text fields returned as typed;
    - anything else (tuples, None, lists) is a text field parsed with
      ``ast.literal_eval``, falling back to the raw string.

    ``values()`` returns a kwargs dict and must be called on the event loop,
    never from a worker thread.
    """

    def __init__(
        self,
        spec=None,
        *,
        choices=None,
        on_change=None,
        empty_text="No parameters.",
    ):
        super().__init__()
        self.classes("w-full gap-1")
        self._rows = {}
        self._on_change = on_change
        self._empty_text = empty_text
        self.set_spec(spec or [], choices=choices)

    def set_spec(self, spec, choices=None):
        choices = choices or {}
        self.clear()
        self._rows = {}
        changed = (lambda _e: self._on_change()) if self._on_change else None
        with self:
            if not spec:
                ui.label(self._empty_text).classes("text-grey")
                return
            with ui.grid(
                columns="repeat(auto-fill, minmax(14rem, 1fr))"
            ).classes("w-full gap-x-4 gap-y-1 items-center"):
                for name, default in spec:
                    label = name.replace("_", " ")
                    if name in choices:
                        options = list(choices[name])
                        if default is not None and default not in options:
                            options.insert(0, default)
                        el = ui.select(
                            options,
                            value=default,
                            label=label,
                            on_change=changed,
                        ).props("dense")
                        kind = "choice"
                    elif isinstance(default, bool):
                        el = ui.switch(label, value=default, on_change=changed)
                        kind = "bool"
                    elif isinstance(default, (int, float)):
                        el = ui.number(
                            label, value=default, on_change=changed
                        ).props("dense")
                        kind = "int" if isinstance(default, int) else "float"
                    elif isinstance(default, str):
                        el = ui.input(
                            label, value=default, on_change=changed
                        ).props("dense")
                        kind = "str"
                    else:
                        el = ui.input(
                            label,
                            value="" if default is None else repr(default),
                            on_change=changed,
                        ).props("dense")
                        kind = "literal"
                    self._rows[name] = (kind, el, default)

    def values(self):
        out = {}
        for name, (kind, el, default) in self._rows.items():
            if kind == "bool":
                out[name] = bool(el.value)
            elif kind in ("int", "float"):
                if el.value is None:
                    out[name] = default
                else:
                    out[name] = (
                        int(el.value) if kind == "int" else float(el.value)
                    )
            elif kind in ("choice", "str"):
                out[name] = default if el.value in (None, "") else el.value
            else:
                out[name] = parse_value(el.value or "", default)
        return out


# ------------------------------------------------- worker-thread helpers
def render_figure_png(fig, dpi=150):
    """Rasterize a Matplotlib figure to PNG bytes and release it.

    Runs on a worker thread with the Agg backend, so a slow 3D plot never
    blocks the UI.
    """
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)
    return buf.getvalue()


def zip_files(paths):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            zf.write(p, arcname=os.path.basename(p))
    return buf.getvalue()


def waveform_svg(preview, bar_fractions=()):
    """Inline SVG of a min/max waveform envelope (or a placeholder).

    A single filled polygon is far cheaper for the browser than a charting
    library, and ``currentColor`` makes it follow the theme's text color.
    ``bar_fractions`` (0-1 positions) draw faint bar lines for metered audio.
    """
    if preview is None:
        return (
            '<div class="w-full h-full flex items-center justify-center '
            'text-sm text-grey">No audio yet</div>'
        )
    _times, lows, highs = preview
    n = len(highs)
    peak = float(max(abs(lows.min()), abs(highs.max()))) or 1.0
    xs = [1000.0 * i / max(n - 1, 1) for i in range(n)]
    top = [
        f"{x:.1f},{50 - 48 * float(y) / peak:.1f}" for x, y in zip(xs, highs)
    ]
    bottom = [
        f"{x:.1f},{50 - 48 * float(y) / peak:.1f}"
        for x, y in zip(reversed(xs), reversed(lows))
    ]
    bars = "".join(
        f'<line x1="{1000 * f:.1f}" y1="0" x2="{1000 * f:.1f}" y2="100" '
        'stroke="currentColor" stroke-opacity="0.3" stroke-width="1" '
        'stroke-dasharray="3 3" vector-effect="non-scaling-stroke"/>'
        for f in bar_fractions
        if 0 < f < 1
    )
    return (
        '<svg viewBox="0 0 1000 100" preserveAspectRatio="none" '
        'class="w-full h-full" role="img" aria-label="Waveform">'
        '<line x1="0" y1="50" x2="1000" y2="50" stroke="currentColor" '
        'stroke-opacity="0.25" stroke-width="1" vector-effect="non-scaling-stroke"/>'
        + bars
        + f'<polygon points="{" ".join(top + bottom)}" fill="currentColor" '
        'fill-opacity="0.55" stroke="currentColor" stroke-width="0.6" '
        'vector-effect="non-scaling-stroke"/></svg>'
    )
