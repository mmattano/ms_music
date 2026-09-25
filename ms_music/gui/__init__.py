"""Browser-based GUI for ms_music (NiceGUI).

Launch with the ``ms-music-gui`` console script or ``python -m ms_music.gui``.
"""

# Figures are rendered to PNG on worker threads, so pyplot must use the
# non-interactive Agg backend. On macOS the default "MacOSX" backend needs
# the main event loop and hangs when a worker thread calls plt.subplots.
# Force Agg before visualizations imports pyplot.
import matplotlib as _matplotlib

_matplotlib.use("Agg")


def main(argv=None):
    # Imported lazily so ``import ms_music.gui`` stays cheap.
    from .app import main as _main

    _main(argv)


__all__ = ["main"]
