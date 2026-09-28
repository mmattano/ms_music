"""Browser-based GUI for ms_music (NiceGUI).

Launch with the ``ms-music-gui`` console script or ``python -m ms_music.gui``.
"""

DEFAULT_PORT = 8765
PORT_RANGE = range(DEFAULT_PORT, DEFAULT_PORT + 10)


def running_instance():
    """URL of an ms_music GUI already serving on this machine, or None.

    Uses only the standard library, so a second launch finds the first one
    in well under a second, before importing the heavy GUI stack.
    """
    import json
    import urllib.request

    for port in PORT_RANGE:
        url = f"http://127.0.0.1:{port}"
        try:
            with urllib.request.urlopen(
                f"{url}/ms_music/ping", timeout=0.3
            ) as r:
                if json.loads(r.read().decode()).get("app") == "ms_music":
                    return url
        except Exception:
            continue
    return None


def main(argv=None):
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if "--port" not in args and "-h" not in args and "--help" not in args:
        existing = running_instance()
        if existing:
            # Already running (e.g. the icon was double-clicked twice):
            # just show it.
            print(f"ms_music is already running at {existing}")
            if "--no-browser" not in args:
                import webbrowser

                webbrowser.open(existing)
            return

    # Imported lazily so ``import ms_music.gui`` stays cheap.
    from .app import main as _main

    _main(args)


def log_path():
    """Where the desktop launcher writes its log."""
    import os
    import sys

    if sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Logs/ms_music")
    elif os.name == "nt":
        base = os.path.join(
            os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"),
            "ms_music",
        )
    else:
        base = os.path.join(os.path.expanduser("~"), ".cache", "ms_music")
    return os.path.join(base, "gui.log")


def main_app(argv=None):
    """Desktop-shortcut entry point (``ms-music``).

    Opens the GUI in the browser and quits a minute after the last tab
    closes. There's no console, so all output goes to :func:`log_path`.
    """
    import os
    import sys

    path = log_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 5_000_000:
        os.replace(path, path + ".old")
    log = open(path, "a", buffering=1, encoding="utf-8")
    # Windowed launches have no console: sys.stdout/stderr may be None,
    # which breaks print() callers and progress bars.
    sys.stdout = sys.stderr = log
    import time

    print(f"\n=== ms_music started {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    try:
        main(
            [
                "--quit-when-closed",
                *(argv if argv is not None else sys.argv[1:]),
            ]
        )
    except SystemExit:
        raise
    except BaseException:
        import traceback

        traceback.print_exc()
        raise


__all__ = ["main", "main_app", "log_path", "running_instance"]
