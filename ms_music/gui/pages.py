"""The GUI's pages. Each ``*_page(shell)`` builds one sidebar page.

Pages talk to the rest of the app only through :class:`Shell`: the engine
(``shell.c``), shared state, and the task runner. Handlers follow the app's
threading rule: read widget values on the event loop, then hand plain values
to ``shell.run_task`` so engine work runs on a worker thread.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List

from nicegui import ui

from ..midi_generator import MusicMeter, QuantizationMode
from . import catalog
from .components import (
    LocalFilePicker,
    ParamForm,
    render_figure_png,
    zip_files,
)
from .controller import (
    CURRENT,
    FREQUENCY_MAPPINGS,
    ROOT_NOTES,
    function_param_spec,
)

INPUT_EXTENSIONS = {"mzML": [".mzml"], "FID": [], "WAV": [".wav"]}

METERS = {m.name: f"{m.beats_per_measure}/{m.note_value}" for m in MusicMeter}
SUBDIVISIONS = {4: "Quarter notes", 8: "Eighth notes", 16: "Sixteenth notes",
                32: "32nd notes"}
FEELS = {"strict_grid": "Straight", "swing": "Swing", "humanized": "Humanized"}
AGGREGATES = {"max": "Loudest scan", "mean": "Average", "sum": "Sum"}


@dataclass
class Shell:
    """What pages may use from the app."""

    c: Any                                   # SonifierController
    state: Any                               # AppState
    action: Callable[..., ui.button]         # button disabled while busy
    run_task: Callable                       # async (desc, fn, updates_audio)
    refresh_player: Callable[[], None]
    _version_listeners: List[Callable[[], None]] = field(default_factory=list)
    _audio_listeners: List[Callable[[], None]] = field(default_factory=list)

    def on_versions_changed(self, callback):
        self._version_listeners.append(callback)

    def versions_changed(self):
        for callback in self._version_listeners:
            callback()

    def on_audio_changed(self, callback):
        self._audio_listeners.append(callback)

    def audio_changed(self):
        for callback in self._audio_listeners:
            callback()


# ----------------------------------------------------------------- helpers
def num(label, value, **kwargs):
    return ui.number(label, value=value, **kwargs).props("dense").classes("w-40")


def section(title, caption=None):
    card = ui.card().classes("w-full")
    with card:
        ui.label(title).classes("text-sm font-semibold text-grey-7")
        if caption:
            ui.label(caption).classes("text-xs text-grey -mt-2")
    return card


def labeled_slider(label, value, min, max, step):
    with ui.column().classes("w-48 gap-0"):
        caption = ui.label().classes("text-xs text-grey-7")
        slider = ui.slider(min=min, max=max, step=step, value=value).props(
            "label dense")
        caption.bind_text_from(slider, "value",
                               lambda v: f"{label}: {v:.2f}")
        return slider


def version_picker(shell, label="Audio versions"):
    """Multi-select over the kept snapshots plus the current audio."""
    picker = ui.select([], multiple=True, value=[], label=label).props(
        "use-chips dense").classes("min-w-[22rem]")

    def refresh():
        options = [CURRENT] + shell.c.snapshot_labels()
        picker.set_options(options,
                           value=[v for v in picker.value if v in options])

    shell.on_versions_changed(refresh)
    refresh()
    return picker


def saved_versions_row(shell):
    """Removable chips for the kept snapshots."""
    row = ui.row().classes("items-center gap-1")

    def remove(label):
        shell.c.remove_snapshot(label)
        shell.versions_changed()

    def refresh():
        row.clear()
        with row:
            labels = shell.c.snapshot_labels()
            if not labels:
                ui.label("No kept versions yet: use the bookmark button in "
                         "the player bar to keep the current audio."
                         ).classes("text-xs text-grey")
            for label in labels:
                ui.chip(label, icon="bookmark", removable=True).props(
                    "dense outline color=primary").on(
                    "remove", lambda _e, lb=label: remove(lb))

    shell.on_versions_changed(refresh)
    refresh()
    return row


# -------------------------------------------------------------------- Data
def data_page(shell):
    c, state = shell.c, shell.state
    with section("Input"):
        input_type = ui.toggle(["mzML", "FID", "WAV"], value="mzML").props("no-caps")
        with ui.row().classes("w-full items-end no-wrap gap-2"):
            file_input = ui.input(
                "File path", placeholder="/path/to/data.mzML").classes(
                "grow").props("clearable")

            async def browse():
                start = (os.path.dirname(file_input.value)
                         if file_input.value else state.last_dir)
                path = await LocalFilePicker(
                    start, INPUT_EXTENSIONS[input_type.value])
                if path:
                    file_input.value = path
                    state.last_dir = os.path.dirname(path)

            ui.button("Browse", icon="folder", on_click=browse).props("outline")

    with section("Settings"):
        with ui.row().classes("gap-4"):
            sample_rate = num("Sample rate (Hz)", 44100, step=1)
            duration_min = num("Duration (min)", 0.5, step=0.1)
            duration_min.bind_visibility_from(
                input_type, "value", lambda v: v != "WAV")
        with ui.row().classes("gap-4") as mzml_opts:
            ms_level = num("MS level", 1, min=1, step=1)
            scan_lo = num("Scan start (0–1)", None, min=0, max=1, step=0.05)
            scan_hi = num("Scan end (0–1)", None, min=0, max=1, step=0.05)
        mzml_opts.bind_visibility_from(input_type, "value", lambda v: v == "mzML")
        with ui.row().classes("gap-4") as fid_opts:
            fid_rate = num("Acquisition rate (Hz)", 1e7)
            fid_conv = num("Conversion factor", 4096)
        fid_opts.bind_visibility_from(input_type, "value", lambda v: v == "FID")

    data_info = ui.label("Choose an input type and file, then Load.").classes(
        "text-grey-7")

    async def do_load():
        path = (file_input.value or "").strip()
        if not path:
            ui.notify("Choose a file first.", type="warning")
            return
        kind = input_type.value
        sr = int(sample_rate.value or 44100)
        dur = float(duration_min.value or 0.5)
        if kind == "mzML":
            lo, hi = scan_lo.value, scan_hi.value
            rng = None if lo is None or hi is None else (float(lo), float(hi))
            level = int(ms_level.value or 1)
            fn = lambda: c.load_mzml(path, level, dur, sr, rng)  # noqa: E731
        elif kind == "FID":
            rate, conv = float(fid_rate.value), float(fid_conv.value)
            fn = lambda: c.load_fid(path, dur, sr, rate, conv)  # noqa: E731
        else:
            fn = lambda: c.load_wav(path, sr)  # noqa: E731
        ok, msg = await shell.run_task(f"Loading {kind}", fn, updates_audio=True)
        if ok:
            data_info.text = msg
            state.last_dir = os.path.dirname(path)
        shell.versions_changed()  # a new sample rate may drop snapshots

    shell.action("Load", icon="upload_file", on_click=do_load)


# ------------------------------------------------------------------ Sonify
def sonify_page(shell):
    c = shell.c
    with section("Synthesis"):
        with ui.row().classes("gap-4"):
            method = ui.select(
                {"gradient": "Gradient (continuous tones)",
                 "adsr": "ADSR (one note per scan/step)"},
                value="gradient", label="Method").classes("w-64")
            mapping = ui.select(FREQUENCY_MAPPINGS, value="inverse_log",
                                label="Frequency mapping").classes("w-48")
            fmin = num("Freq min (Hz)", 200)
            fmax = num("Freq max (Hz)", 4000)
        with ui.row().classes("gap-4"):
            ms2 = ui.select({"none": "None", "dda": "DDA precursor",
                             "dia": "DIA window"},
                            value="none", label="MS2 precursor tones").classes("w-48")
            max_peaks = num("Max peaks / scan", None, min=1, step=1).tooltip(
                "Optional: keep only the N most intense peaks per scan "
                "(recommended for MS2).")

    with section("ADSR envelope") as adsr_card:
        adsr_random = ui.switch("Randomize per note", value=False)
        with ui.row().classes("gap-4"):
            adsr_attack = num("Attack %", 0.1, step=0.05)
            adsr_decay = num("Decay %", 0.1, step=0.05)
            adsr_sustain = num("Sustain level", 0.7, step=0.05)
            adsr_release = num("Release %", 0.2, step=0.05)
        for el in (adsr_attack, adsr_decay, adsr_sustain, adsr_release):
            el.bind_enabled_from(adsr_random, "value", lambda v: not v)
    adsr_card.bind_visibility_from(method, "value", lambda v: v == "adsr")

    with section("Pitch", "Continuous pitch by default; optionally snap every "
                          "tone to a scale."):
        snap = ui.switch("Snap to a musical scale", value=False)
        with ui.column().classes("gap-2") as pitch_opts:
            with ui.row().classes("gap-4 items-center"):
                scale = ui.select(c.all_scales(), value="major", label="Scale",
                                  with_input=True).classes("w-56")
                root = ui.select(ROOT_NOTES, value="C", label="Root").classes("w-24")
                tuning = num("Tuning A4 (Hz)", 440.0)
            with ui.row().classes("gap-4 items-center"):
                edo = num("EDO divisions", 12, min=1, step=1)
                just = ui.switch("Just intonation", value=False)
                log_dist = ui.switch("Nearest note in log pitch", value=True)
            ui.label("Pair an EDO scale (e.g. 19_edo_diatonic) with matching EDO "
                     "divisions, or a just_* scale with Just intonation on."
                     ).classes("text-xs text-grey")
        pitch_opts.bind_visibility_from(snap, "value")

    with section("Rhythm", "Equal time per scan by default; optionally regroup "
                           "scans onto a beat grid."):
        metered = ui.switch("Play in meter", value=False)
        with ui.column().classes("gap-2 w-full") as rhythm_opts:
            with ui.row().classes("gap-4 items-center"):
                meter = ui.select(METERS, value="FOUR_FOUR",
                                  label="Meter").classes("w-28")
                tempo = num("Tempo (quarter BPM)", 120, min=20, max=400, step=1)
                subdivision = ui.select(SUBDIVISIONS, value=16,
                                        label="Grid step").classes("w-44")
                feel = ui.select(FEELS, value="strict_grid",
                                 label="Feel").classes("w-36")
                aggregate = ui.select(AGGREGATES, value="max",
                                      label="Scans per step").classes("w-40")
            with ui.row().classes("gap-8 items-end"):
                threshold = labeled_slider("Note threshold", 0.05, 0, 0.5, 0.01)
                accent = labeled_slider("Downbeat accent", 0.4, 0, 1, 0.05)
                gate = labeled_slider("Gap before next note (gate)",
                                      0.9, 0.1, 1, 0.05)
                with ui.column().classes("gap-0") as swing_box:
                    swing = labeled_slider("Swing ratio", 0.67, 0.5, 0.8, 0.01)
                swing_box.bind_visibility_from(feel, "value",
                                               lambda v: v == "swing")
            bar_hint = ui.label().classes("text-xs text-grey")
        rhythm_opts.bind_visibility_from(metered, "value")

    def update_bar_hint():
        m = MusicMeter[meter.value]
        bpm = float(tempo.value or 120)
        bar_s = m.beats_per_measure * (4.0 / m.note_value) * 60.0 / bpm
        steps = m.beats_per_measure * int(subdivision.value) // m.note_value
        warn = ("" if int(subdivision.value) >= m.note_value else
                f" — use eighth notes or finer for {METERS[meter.value]}")
        bar_hint.text = (f"One bar = {bar_s:.2f} s, {steps} steps; the length "
                         f"snaps to whole bars. Notes start and end on grid "
                         f"steps and last as long as each pitch's signal "
                         f"stays above the threshold.{warn}")

    for el in (meter, tempo, subdivision):
        el.on_value_change(lambda _e: update_bar_hint())
    update_bar_hint()

    async def do_sonify():
        kw = dict(
            method=method.value,
            frequency_mapping=mapping.value,
            freq_range=(float(fmin.value), float(fmax.value)),
            ms2_mode=None if ms2.value == "none" else ms2.value,
            max_peaks_per_scan=int(max_peaks.value) if max_peaks.value else None,
        )
        if method.value == "adsr":
            kw["adsr_settings"] = {
                "randomize": bool(adsr_random.value),
                "attack_time_pc": float(adsr_attack.value),
                "decay_time_pc": float(adsr_decay.value),
                "sustain_level_pc": float(adsr_sustain.value),
                "release_time_pc": float(adsr_release.value),
            }
        if snap.value:
            kw.update(
                scale=scale.value, root_note=root.value,
                tuning_freq=float(tuning.value),
                edo_divisions=int(edo.value or 12),
                use_just_intonation=bool(just.value),
                use_log_distance=bool(log_dist.value),
            )
        if metered.value:
            kw["rhythm"] = dict(
                meter=meter.value, tempo=float(tempo.value or 120),
                subdivision=int(subdivision.value), mode=feel.value,
                swing_ratio=float(swing.value), accent=float(accent.value),
                gate=float(gate.value), aggregate=aggregate.value,
                note_threshold=float(threshold.value),
            )
        await shell.run_task("Generating audio", lambda: c.sonify(**kw),
                             updates_audio=True)

    shell.action("Generate audio", icon="play_circle", on_click=do_sonify)


# ----------------------------------------------------------------- Effects
def effects_page(shell):
    c = shell.c
    effects = c.list_effects()
    with section("Effect"):
        effect_select = ui.select(effects, value=effects[0] if effects else None,
                                  label="Effect", with_input=True).classes("w-64")
        effect_form = ParamForm(empty_text="This effect has no parameters.")
        effect_select.on_value_change(
            lambda e: effect_form.set_spec(c.effect_param_spec(e.value))
            if e.value else None)
        if effects:
            effect_form.set_spec(c.effect_param_spec(effects[0]))

    async def do_apply():
        name = effect_select.value
        if not name:
            return
        params = effect_form.values()
        await shell.run_task(f"Applying {name}",
                             lambda: c.apply_effect(name, params),
                             updates_audio=True)

    async def do_undo():
        await shell.run_task("Undoing effect", c.undo_effect, updates_audio=True)

    async def do_reset():
        await shell.run_task("Resetting effects", c.reset_effects,
                             updates_audio=True)

    with ui.row().classes("gap-2"):
        shell.action("Apply effect", icon="add", on_click=do_apply)
        shell.action("Undo last", icon="undo", on_click=do_undo).props("outline")
        shell.action("Reset to base", icon="restart_alt",
                     on_click=do_reset).props("outline")
    ui.label("The applied chain is shown in the player bar below.").classes(
        "text-xs text-grey")


# -------------------------------------------------------------------- MIDI
def midi_page(shell):
    c = shell.c
    with section("Musical system"):
        with ui.row().classes("gap-4"):
            m_scale = ui.select(c.all_scales(), value="major", label="Scale",
                                with_input=True).classes("w-56")
            m_root = ui.select(ROOT_NOTES, value="C", label="Root").classes("w-24")
            m_meter = ui.select(METERS, value="FOUR_FOUR",
                                label="Meter").classes("w-28")
            m_qmode = ui.select({q.name: q.name.replace("_", " ").title()
                                 for q in QuantizationMode},
                                value="STRICT_GRID",
                                label="Quantization").classes("w-44")
        with ui.row().classes("gap-4 items-center"):
            m_tempo = num("Tempo (BPM)", 120, min=1, step=1)
            m_edo = num("EDO divisions", 12, min=1, step=1)
            m_just = ui.switch("Just intonation", value=False)
    with section("Peak detection"):
        with ui.row().classes("gap-4"):
            m_duration = num("Duration (s)", 60.0)
            m_percentile = num("Intensity percentile", 90.0, min=0, max=100)
            m_maxpeaks = num("Max peaks / scan", 500, min=1, step=1)
            m_mapping = ui.select(FREQUENCY_MAPPINGS, value="inverse_log",
                                  label="Mapping").classes("w-48")
    with section("Clustering & output"):
        with ui.row().classes("gap-4 items-center"):
            m_cluster = ui.switch("m/z clustering", value=False)
            m_ppm = num("Tolerance (ppm)", 20.0)
            m_ppm.bind_visibility_from(m_cluster, "value")
        with ui.row().classes("gap-4 items-center"):
            m_instrument = num("GM instrument (1–128)", 1, min=1, max=128, step=1)
            m_maxnotes = num("Max simultaneous", 8, min=1, step=1)
            m_voices = num("Voices (1–4)", 1, min=1, max=4, step=1)
            m_separate = ui.switch("Separate file per voice", value=False)
            m_notecsv = ui.switch("Export note CSV", value=False)

    midi_info = ui.label("Requires a loaded mzML file.").classes("text-grey-7")

    async def do_midi():
        stem = os.path.splitext(os.path.basename(c.source_path or "ms_music"))[0]
        out_dir = tempfile.mkdtemp(prefix="midi_", dir=shell.state.work_dir)
        out_path = os.path.join(out_dir, f"{stem}.mid")
        kw = dict(
            output_path=out_path,
            scale=m_scale.value, root_note=m_root.value,
            tempo=int(m_tempo.value), meter=MusicMeter[m_meter.value],
            quantization_mode=QuantizationMode[m_qmode.value],
            edo_divisions=int(m_edo.value or 12),
            use_just_intonation=bool(m_just.value),
            duration_seconds=float(m_duration.value),
            enable_mz_clustering=bool(m_cluster.value),
            mz_tolerance_ppm=float(m_ppm.value),
            intensity_threshold_percentile=float(m_percentile.value),
            max_peaks_per_scan=int(m_maxpeaks.value),
            frequency_mapping=m_mapping.value,
            instrument=int(m_instrument.value),
            max_simultaneous_notes=int(m_maxnotes.value),
            export_note_data=bool(m_notecsv.value),
            num_voices=int(m_voices.value or 1),
            separate_files=bool(m_separate.value),
        )

        def job():
            message, paths = c.generate_midi(**kw)
            if len(paths) == 1:
                with open(paths[0], "rb") as fh:
                    return message, fh.read(), os.path.basename(paths[0])
            return message, zip_files(paths), f"{stem}_midi.zip"

        ok, result = await shell.run_task("Generating MIDI", job)
        if ok:
            message, payload, filename = result
            midi_info.text = message
            ui.download.content(payload, filename)

    shell.action("Generate & download MIDI", icon="download", on_click=do_midi)


# ---------------------------------------------------- shared entry picker
def _entry_controls(shell, entries, groups=None, form_on_change=None):
    """Group toggle + entry select + versions + auto-generated form.

    Returns a namespace with the widgets and ``current()`` / ``reason()``.
    """
    c = shell.c
    ns = type("EntryControls", (), {})()
    labels_in = (lambda g: [e.label for e in entries if e.group == g])
    with ui.row().classes("items-end gap-3 w-full"):
        if groups:
            ns.group = ui.toggle(groups, value=groups[0]).props("no-caps")
            first = labels_in(groups[0])
        else:
            ns.group = None
            first = [e.label for e in entries]
        ns.select = ui.select(first, value=first[0], label="Show").classes("w-72")
    with ui.column().classes("w-full gap-1") as versions_box:
        with ui.row().classes("items-center gap-3 w-full"):
            ns.versions = version_picker(shell)
            ui.label("Compare kept versions with each other or with the "
                     "current audio.").classes("text-xs text-grey")
        saved_versions_row(shell)
    ns.help = ui.label().classes("text-xs text-grey")
    ns.form = ParamForm(empty_text="No options for this one.",
                        on_change=form_on_change)
    ns.reason = ui.label().classes("text-sm text-warning")

    def current():
        return catalog.find(entries, ns.select.value)

    def reason():
        entry = current()
        return catalog.unmet_need(
            entry,
            has_audio=c.current_audio() is not None,
            has_ms_data=c.has_ms_data(),
            n_versions=len(c.audio_versions(ns.versions.value)),
        )

    def on_entry():
        entry = current()
        ns.form.set_spec(
            function_param_spec(entry.func, skip=catalog.skip_names(),
                                overrides=entry.defaults),
            choices=entry.choices)
        ns.help.text = entry.help
        ns.help.visible = bool(entry.help)
        uses_versions = bool({catalog.COMPARE, catalog.PAIR} & set(entry.needs)) \
            or entry.label == "Summary grid"
        versions_box.visible = uses_versions
        ns.reason.text = ""
        if form_on_change:
            form_on_change()

    def on_group(e):
        options = labels_in(e.value)
        ns.select.set_options(options, value=options[0])

    if ns.group is not None:
        ns.group.on_value_change(on_group)
    ns.select.on_value_change(lambda _e: on_entry())
    on_entry()
    ns.current, ns.reason_text = current, reason
    return ns


# --------------------------------------------------------------- Visualize
def visualize_page(shell):
    c, state = shell.c, shell.state
    ctl = _entry_controls(shell, catalog.PLOTS, groups=catalog.PLOT_GROUPS)

    async def do_render():
        entry = ctl.current()
        why = ctl.reason_text()
        ctl.reason.text = why or ""
        if why:
            ui.notify(why, type="warning")
            return
        params = ctl.form.values()
        labels = list(ctl.versions.value)
        ok, png = await shell.run_task(
            f"Rendering {entry.label}",
            lambda: render_figure_png(c.render_plot(entry, params, labels)))
        if ok:
            state.figure_png, state.figure_name = png, entry.label
            state.figure_version += 1
            show_figure()

    def download_figure():
        if state.figure_png is None:
            ui.notify("Render a plot first.", type="warning")
            return
        fname = "".join(ch if ch.isalnum() else "_"
                        for ch in state.figure_name.lower()).strip("_") + ".png"
        ui.download.content(state.figure_png, fname, "image/png")

    with ui.row().classes("gap-2"):
        shell.action("Render", icon="insights", on_click=do_render)
        ui.button("Download PNG", icon="download",
                  on_click=download_figure).props("outline")
    figure_box = ui.column().classes("w-full items-center")

    def show_figure():
        figure_box.clear()
        with figure_box:
            if state.figure_png is None:
                ui.label("No plot yet.").classes("text-grey py-16")
            else:
                ui.image(f"/ms_music/figure/{state.figure_version}.png").classes(
                    "w-full max-h-[calc(100vh-18rem)] rounded").props(
                    "fit=contain no-spinner")

    show_figure()


# ------------------------------------------------------------------- Video
def video_page(shell):
    c, state = shell.c, shell.state
    if shutil.which("ffmpeg") is None:
        with ui.card().classes("w-full bg-warning text-black"):
            ui.label("Video export needs ffmpeg, which wasn't found on your "
                     "PATH.").classes("font-semibold")
            ui.label("Install it (macOS: brew install ffmpeg · Ubuntu: sudo apt "
                     "install ffmpeg · Windows: winget install ffmpeg), then "
                     "restart ms-music-gui.")
        return

    estimate = ui.label().classes("text-xs text-grey")
    ctl = None  # assigned below; the form calls update_estimate while building

    def update_estimate():
        if ctl is None:
            return
        entry = ctl.current()
        params = ctl.form.values()
        audio_s = c.duration_seconds()
        if not audio_s and not params.get("duration_seconds"):
            estimate.text = "Generate or load audio to see a time estimate."
            return
        frames = catalog.estimated_frames(entry, params, audio_s)
        seconds = catalog.estimated_seconds(entry, params, audio_s)
        when = (f"about {seconds / 60:.0f} min" if seconds >= 90
                else f"about {max(seconds, 1):.0f} s")
        estimate.text = (f"≈ {frames:,} frames, {when} to render (measured on a "
                         f"busy laptop; lower fps, dpi or duration to go "
                         f"faster).")

    ctl = _entry_controls(shell, catalog.VIDEOS, form_on_change=update_estimate)
    estimate.move(target_index=-1)  # below the form
    update_estimate()
    shell.on_audio_changed(update_estimate)

    progress_state = {"frame": 0, "total": 0, "t0": 0.0}

    def on_progress(frame, total):  # worker thread: plain dict writes only
        progress_state["frame"], progress_state["total"] = frame + 1, total

    async def do_render():
        entry = ctl.current()
        why = ctl.reason_text()
        ctl.reason.text = why or ""
        if why:
            ui.notify(why, type="warning")
            return
        params = ctl.form.values()
        labels = list(ctl.versions.value)
        out = os.path.join(state.work_dir, f"video_{state.video_version + 1}.mp4")
        state.cancel_video = False
        progress_state.update(frame=0, total=0, t0=time.monotonic())
        progress_row.visible = True
        cancel_btn.enabled = True
        ticker.activate()
        try:
            ok, path = await shell.run_task(
                f"Rendering {entry.label}",
                lambda: c.render_video(entry, params, out, labels,
                                       progress=on_progress,
                                       cancelled=lambda: state.cancel_video))
        finally:
            ticker.deactivate()
            progress_row.visible = False
            cancel_btn.enabled = False
        if ok:
            state.video_path, state.video_name = path, entry.label
            state.video_version += 1
            show_video()

    def tick():
        frame, total = progress_state["frame"], progress_state["total"]
        elapsed = time.monotonic() - progress_state["t0"]
        if not total:
            bar.value = 0
            progress_label.text = f"Preparing… {elapsed:.0f} s"
            return
        bar.value = frame / total
        left = elapsed / max(frame, 1) * (total - frame)
        progress_label.text = (f"Frame {frame:,} / {total:,} · {elapsed:.0f} s "
                               f"elapsed · about {left:.0f} s left")

    def cancel():
        state.cancel_video = True
        progress_label.text = "Cancelling after the current frame…"

    with ui.row().classes("gap-2 items-center"):
        shell.action("Render video", icon="movie", on_click=do_render)
        cancel_btn = ui.button("Cancel", icon="close", on_click=cancel).props(
            "outline color=negative")
        cancel_btn.enabled = False
    with ui.column().classes("w-full gap-1") as progress_row:
        bar = ui.linear_progress(value=0, show_value=False).props("rounded")
        progress_label = ui.label().classes("text-xs text-grey")
    progress_row.visible = False
    ticker = ui.timer(0.3, tick, active=False)

    video_box = ui.column().classes("w-full items-center")

    def download_video():
        if not state.video_path:
            ui.notify("Render a video first.", type="warning")
            return
        name = "".join(ch if ch.isalnum() else "_"
                       for ch in state.video_name.lower()).strip("_")
        ui.download.file(state.video_path, f"{name}.mp4")

    def show_video():
        video_box.clear()
        with video_box:
            if not state.video_path:
                ui.label("No video yet.").classes("text-grey py-10")
                return
            ui.video(f"/ms_music/video/{state.video_version}.mp4").classes(
                "w-full max-h-[calc(100vh-20rem)] rounded")
            ui.button("Download MP4", icon="download",
                      on_click=download_video).props("outline")

    show_video()
