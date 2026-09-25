"""Browser-based GUI for the ms_music workflow, built with NiceGUI.

This module is the shell: shared state, HTTP routes for media, the header,
the player bar and the task runner. The pages themselves live in
:mod:`ms_music.gui.pages`.

Threading rule: event handlers read every widget value on the event loop
into plain Python values, then run engine calls *and* figure/video
rendering via ``run.io_bound``. Only finished results (strings, files,
PNG/WAV bytes) touch UI elements. The browser renders the interface itself,
so spinners and page switching stay live even while the engine is busy.
"""

from __future__ import annotations

import argparse
import atexit
import os
import shutil
import socket
import tempfile

from fastapi import HTTPException, Response
from fastapi.responses import FileResponse
from nicegui import app, run, ui

from . import pages
from .components import waveform_svg
from .controller import RenderCancelled, SonifierController


class AppState:
    """Process-wide state. The app is local and single-user, so every
    browser tab shares one engine; per-tab UI is rebuilt from this."""

    def __init__(self):
        self.controller = SonifierController()
        self.busy = False
        # Media lives in temp files so it can be served with HTTP Range
        # support (browsers need it to load and seek audio and video).
        self.work_dir = tempfile.mkdtemp(prefix="ms_music_gui_")
        atexit.register(shutil.rmtree, self.work_dir, ignore_errors=True)
        self.wav_path: str | None = None
        self.audio_version = 0
        self.preview = None
        self.figure_png: bytes | None = None
        self.figure_name = ""
        self.figure_version = 0
        self.video_path: str | None = None
        self.video_name = ""
        self.video_version = 0
        self.cancel_video = False
        self.last_dir: str | None = None


state = AppState()


def _serve_file(path, media_type):
    if path is None or not os.path.exists(path):
        raise HTTPException(status_code=404)
    return FileResponse(path, media_type=media_type)


# Media is served over HTTP rather than as data URLs so multi-MB buffers
# don't travel through the websocket; the version in each path defeats
# caching.
@app.get("/ms_music/audio/{version}.wav")
def _serve_audio(version: int):
    return _serve_file(state.wav_path, "audio/wav")


@app.get("/ms_music/video/{version}.mp4")
def _serve_video(version: int):
    return _serve_file(state.video_path, "video/mp4")


@app.get("/ms_music/figure/{version}.png")
def _serve_figure(version: int):
    return Response(state.figure_png or b"", media_type="image/png")


def _audio_snapshot(version):
    """Worker-thread helper: write the current audio to a versioned WAV file
    and compute its waveform envelope. Returns (path or None, preview)."""
    c = state.controller
    wav = c.current_wav_bytes()
    path = None
    if wav is not None:
        path = os.path.join(state.work_dir, f"audio_{version}.wav")
        with open(path, "wb") as fh:
            fh.write(wav)
    # ~1k buckets is already finer than the footer is wide.
    return path, c.waveform_preview(max_points=1000)


def build_ui():
    c = state.controller
    ui.colors(primary="#4f46e5", secondary="#0ea5e9", accent="#a855f7")
    dark = ui.dark_mode(None)  # follow the OS preference
    action_buttons: list[ui.button] = []

    def action(*args, **kwargs):
        btn = ui.button(*args, **kwargs)
        action_buttons.append(btn)
        return btn

    # ------------------------------------------------------------- chrome
    with ui.header().classes("items-center justify-between py-2 px-4"):
        with ui.row().classes("items-center gap-2"):
            ui.icon("graphic_eq", size="md")
            ui.label("ms_music").classes("text-xl font-semibold")
            ui.label("Mass spectrometry sonification").classes(
                "text-sm opacity-80 max-sm:hidden")
        with ui.row().classes("items-center gap-1"):
            spinner = ui.spinner("audio", size="md", color="white")
            spinner.visible = False
            status = ui.label("Ready").classes("text-sm opacity-90 mr-2")
            theme_btn = ui.button(on_click=lambda: cycle_theme()).props(
                "flat round color=white").tooltip("Theme: auto / dark / light")
            ui.button(icon="receipt_long", on_click=lambda: log_drawer.toggle()).props(
                "flat round color=white").tooltip("Show log")

    # Cycle auto (follow OS) -> dark -> light; the icon shows the mode.
    theme_icons = {None: "brightness_auto", True: "dark_mode", False: "light_mode"}

    def cycle_theme():
        dark.value = {None: True, True: False, False: None}[dark.value]
        theme_btn.props(f"icon={theme_icons[dark.value]}")

    theme_btn.props(f"icon={theme_icons[dark.value]}")

    with ui.right_drawer(value=False).classes("p-2") as log_drawer:
        ui.label("Log").classes("text-sm font-semibold")
        log = ui.log(max_lines=500).classes("w-full h-[80vh] text-xs")

    # --------------------------------------------------------- player bar
    # Quasar marks dark mode with body--dark (Tailwind's dark: variant
    # doesn't see it), so theme the player bar with plain CSS.
    ui.add_css("""
        .ms-player { background: #fff; color: #111; border-top: 1px solid #e5e7eb; }
        .body--dark .ms-player { background: #1d1d1d; color: #eee; border-color: #333; }
    """)
    with ui.footer().classes("p-0 ms-player"):
        with ui.column().classes("w-full px-4 py-2 gap-1"):
            wave_view = ui.html(waveform_svg(None), sanitize=False).classes(
                "w-full h-20 text-primary")
            with ui.row().classes("w-full items-center gap-3 no-wrap"):
                player = ui.audio("").classes("grow h-10")
                audio_info = ui.label("No audio loaded.").classes(
                    "text-sm text-grey-7 whitespace-nowrap max-md:hidden")
                chain_row = ui.row().classes("gap-1 max-lg:hidden")
                keep_btn = ui.button(icon="bookmark_add",
                                     on_click=lambda: keep_version()).props(
                    "flat round").tooltip("Keep this version for comparison")
                download_wav = ui.button(icon="download",
                                         on_click=lambda: _download_wav()).props(
                    "flat round").tooltip("Download WAV")

    def refresh_player():
        audio_ok = state.wav_path is not None
        duration = c.duration_seconds()
        grid = c.rhythm_grid()
        bars = (grid.bar_times / duration) if grid is not None and duration else ()
        wave_view.content = waveform_svg(
            state.preview if audio_ok else None, bar_fractions=bars)
        if audio_ok:
            player.set_source(f"/ms_music/audio/{state.audio_version}.wav")
            info = f"{duration:.1f} s · {c.sample_rate} Hz"
            if grid is not None:
                info += f" · {grid.describe()}"
            audio_info.text = info
        else:
            player.set_source("")
            audio_info.text = "No audio loaded."
        download_wav.enabled = keep_btn.enabled = audio_ok and not state.busy
        chain_row.clear()
        with chain_row:
            for label in c.chain_labels():
                ui.chip(label, icon="tune").props("dense outline color=primary")

    def _download_wav():
        if state.wav_path is None:
            ui.notify("Nothing to download yet.", type="warning")
            return
        stem = os.path.splitext(os.path.basename(c.source_path or "ms_music"))[0]
        ui.download.file(state.wav_path, f"{stem}.wav")

    async def keep_version():
        if state.wav_path is None:
            ui.notify("Generate or load audio first.", type="warning")
            return
        with ui.dialog() as dialog, ui.card().classes("w-96"):
            ui.label("Keep this version for comparison").classes(
                "text-base font-semibold")
            name = ui.input("Name", value=c.default_snapshot_label()).classes(
                "w-full").on("keydown.enter", lambda: dialog.submit(name.value))
            with ui.row().classes("w-full justify-end"):
                ui.button("Cancel", on_click=lambda: dialog.submit(None)).props("flat")
                ui.button("Keep", on_click=lambda: dialog.submit(name.value))
        result = await dialog
        dialog.delete()
        if result is None:
            return
        label = c.keep_snapshot(result)
        ui.notify(f"Kept “{label}” for comparison.", type="positive")
        write_log(f"Kept version: {label}")
        shell.versions_changed()

    def set_busy(busy, text="Ready"):
        state.busy = busy
        spinner.visible = busy
        status.text = text
        for b in action_buttons:
            b.enabled = not busy
        download_wav.enabled = keep_btn.enabled = (
            (not busy) and state.wav_path is not None)

    def write_log(text):
        log.push(text)

    async def run_task(description, fn, *, updates_audio=False):
        """Run ``fn`` off the event loop. Returns (ok, result)."""
        if state.busy:
            ui.notify("Please wait for the current task to finish.")
            return False, None
        set_busy(True, f"{description}…")

        version = state.audio_version + 1

        def job():
            result = fn()
            snapshot = _audio_snapshot(version) if updates_audio else None
            return result, snapshot

        try:
            result, snapshot = await run.io_bound(job)
        except RenderCancelled:
            write_log(f"Cancelled: {description}")
            ui.notify("Cancelled.", type="info")
            set_busy(False)
            return False, None
        except Exception as exc:  # noqa: BLE001
            write_log(f"ERROR ({description}): {exc}")
            ui.notify(str(exc), type="negative", multi_line=True, timeout=8000)
            set_busy(False, "Error")
            return False, None
        if snapshot is not None:
            old_path = state.wav_path
            state.wav_path, state.preview = snapshot
            state.audio_version = version
            if old_path and old_path != state.wav_path:
                try:
                    os.remove(old_path)
                except OSError:
                    pass
        set_busy(False)
        message = result[0] if isinstance(result, tuple) else result
        if isinstance(message, str):
            write_log(message)
            ui.notify(message, type="positive")
        elif description:
            write_log(f"Done: {description}")
        if updates_audio:
            refresh_player()
            shell.audio_changed()
        return True, result

    shell = pages.Shell(c=c, state=state, action=action, run_task=run_task,
                        refresh_player=refresh_player)

    # --------------------------------------------------------------- pages
    page_list = [
        ("Data", "folder_open", pages.data_page),
        ("Sonify", "waves", pages.sonify_page),
        ("Effects", "tune", pages.effects_page),
        ("MIDI", "music_note", pages.midi_page),
        ("Visualize", "insights", pages.visualize_page),
        ("Video", "movie", pages.video_page),
    ]
    with ui.left_drawer(value=True, bordered=True).props("width=190").classes("p-0"):
        with ui.tabs().props("vertical inline-label no-caps align=left").classes(
                "w-full") as tabs:
            tab_objs = [ui.tab(name, icon=icon) for name, icon, _ in page_list]

    with ui.tab_panels(tabs, value=tab_objs[0]).props("animated=false").classes(
            "w-full max-w-5xl mx-auto bg-transparent"):
        for tab, (_name, _icon, build) in zip(tab_objs, page_list):
            with ui.tab_panel(tab).classes("gap-4"):
                build(shell)

    # Restore shared state for a reloaded/new browser tab.
    refresh_player()
    set_busy(state.busy, status.text)


def _free_port(preferred=8765):
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    return preferred


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="ms-music-gui", description="Launch the ms_music GUI.")
    parser.add_argument("--native", action="store_true",
                        help="open in a desktop window (requires pywebview)")
    parser.add_argument("--port", type=int, default=None,
                        help="port to serve on (default: first free from 8765)")
    parser.add_argument("--no-browser", action="store_true",
                        help="don't open a browser tab automatically")
    args = parser.parse_args(argv)

    native = args.native
    if native:
        try:
            import webview  # noqa: F401
        except ImportError:
            print("--native needs pywebview (pip install pywebview); "
                  "opening in the browser instead.")
            native = False

    ui.run(
        build_ui,
        title="ms_music",
        host="127.0.0.1",
        port=args.port or _free_port(),
        reload=False,
        show=not args.no_browser,
        native=native,
        window_size=(1280, 900) if native else None,
        favicon="🎵",
        dark=None,
        # Keep a tab's server-side state through short disconnects (laptop
        # sleep, a busy browser) instead of dropping it after NiceGUI's 3 s
        # default and forcing a page reload that loses unsaved form values.
        reconnect_timeout=60.0,
    )


if __name__ == "__main__":
    main()
