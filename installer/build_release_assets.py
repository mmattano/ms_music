"""Build the installer downloads for a GitHub release.

    python installer/build_release_assets.py [output_dir]

Writes (default ``dist/installers``):

- ``ms_music-macOS-installer.zip`` with "Install ms_music.command" and
  "Uninstall ms_music.command". Zipped because browsers drop the execute
  permission of downloaded files, while Archive Utility keeps it.
- "Install ms_music.bat" and "Uninstall ms_music.bat": the PowerShell
  scripts embedded in a batch file, so a double-click runs them without
  changing the execution policy.
"""

import os
import stat
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))

BAT_HEADER = (
    "@echo off\r\n"
    "rem {title}: runs the PowerShell script embedded below this line.\r\n"
    "powershell -NoProfile -ExecutionPolicy Bypass -Command "
    "\"$s = Get-Content -LiteralPath '%~f0' -Raw; "
    "$i = $s.IndexOf('#' + 'POWERSHELL'); "
    'Invoke-Expression $s.Substring($i)"\r\n'
    "exit /b %errorlevel%\r\n"
    "#POWERSHELL\r\n"
)


def _read(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as fh:
        return fh.read()


def windows_bat(ps1_name, title):
    script = _read(ps1_name)
    try:
        script.encode("ascii")
    except UnicodeEncodeError as exc:
        # Windows PowerShell 5.1 reads the file as ANSI.
        raise SystemExit(f"{ps1_name} must be ASCII-only: {exc}")
    body = script.replace("\r\n", "\n").replace("\n", "\r\n")
    return BAT_HEADER.format(title=title) + body


def macos_zip(path):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for source, name in (
            ("install_macos.sh", "Install ms_music.command"),
            ("uninstall_macos.sh", "Uninstall ms_music.command"),
        ):
            info = zipfile.ZipInfo(
                f"ms_music installer/{name}", date_time=time.localtime()[:6]
            )
            info.create_system = 3  # unix, so the mode bits are honoured
            info.external_attr = (stat.S_IFREG | 0o755) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, _read(source))


def main(out_dir=None):
    out_dir = out_dir or os.path.join(HERE, "..", "dist", "installers")
    os.makedirs(out_dir, exist_ok=True)
    macos_zip(os.path.join(out_dir, "ms_music-macOS-installer.zip"))
    for ps1, name in (
        ("install_windows.ps1", "Install ms_music.bat"),
        ("uninstall_windows.ps1", "Uninstall ms_music.bat"),
    ):
        with open(
            os.path.join(out_dir, name), "w", encoding="ascii", newline=""
        ) as fh:
            fh.write(windows_bat(ps1, name[:-4]))
    print("Wrote installers to", os.path.abspath(out_dir))
    for name in sorted(os.listdir(out_dir)):
        print("  ", name)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
