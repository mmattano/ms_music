"""Small helpers shared across the package."""

import shutil
from typing import Optional


def ffmpeg_path() -> Optional[str]:
    """Path of an ffmpeg executable for video export.

    Prefers an ffmpeg on the PATH; otherwise uses the binary bundled with
    the ``imageio-ffmpeg`` dependency, so video works without installing
    ffmpeg separately. None if neither is available.
    """
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def configure_matplotlib_ffmpeg() -> Optional[str]:
    """Point matplotlib's animation writer at :func:`ffmpeg_path`."""
    import matplotlib

    path = ffmpeg_path()
    if path:
        matplotlib.rcParams["animation.ffmpeg_path"] = path
    return path
