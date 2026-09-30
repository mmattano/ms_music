#!/bin/bash
# Install or update ms_music for the current user (no admin rights needed).
#
# What it does:
#   1. installs "uv" (a small tool that manages Python) if it isn't there;
#   2. installs ms_music, with its own Python, in an isolated place;
#   3. creates ~/Applications/ms_music.app and a Desktop shortcut.
# Running it again updates ms_music.
#
# Environment (for testing): MS_MUSIC_PACKAGE (install from a wheel/path
# instead of PyPI), MS_MUSIC_SKIP_OPEN=1 (don't open the app at the end).

set -uo pipefail

PACKAGE="${MS_MUSIC_PACKAGE:-ms_music}"
PYTHON_VERSION="3.12"
APP="$HOME/Applications/ms_music.app"

bold() { printf '\n\033[1m%s\033[0m\n' "$1"; }

finish() {
    # Keep the window open when started by double-click (.command).
    if [[ "$0" == *.command ]] && [ -e /dev/tty ]; then
        printf '\nPress any key to close this window…'
        read -r -n 1 _ </dev/tty || true
    fi
    exit "$1"
}

fail() {
    printf '\n\033[31mSorry, something went wrong: %s\033[0m\n' "$1"
    printf 'Please send a screenshot of this window to the ms_music maintainers.\n'
    finish 1
}

bold "Installing ms_music"
echo "This takes a few minutes the first time (it downloads Python and the"
echo "scientific libraries). Nothing is installed system-wide."

bold "Step 1 of 3: getting the installer tool (uv)"
export PATH="$HOME/.local/bin:$PATH"
if command -v uv >/dev/null 2>&1; then
    echo "uv is already installed."
else
    curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh \
        || fail "could not download uv (check your internet connection)."
    command -v uv >/dev/null 2>&1 || fail "uv was not found after installing it."
fi

bold "Step 2 of 3: installing ms_music"
# A running ms_music would keep serving the old version: ask it to quit
# (ports as in ms_music.gui.PORT_RANGE), then make sure it has.
stopped=""
for port in $(seq 8765 8774); do
    curl -sf -m 2 -X POST -H "X-MS-Music: quit" \
        "http://127.0.0.1:$port/ms_music/quit" >/dev/null 2>&1 && stopped=1
done
TOOL_PY="$(uv tool dir 2>/dev/null)/ms-music/bin/python"
pkill -f "$TOOL_PY" >/dev/null 2>&1 && stopped=1
if [ -n "$stopped" ]; then
    echo "Stopped the running ms_music."
    sleep 2
fi
# --compile-bytecode: prepare Python files now, so the first start is quick.
uv tool install --python "$PYTHON_VERSION" --upgrade --force --compile-bytecode "$PACKAGE" \
    || fail "uv could not install ms_music."

BIN_DIR="$(uv tool dir --bin)"
TOOL_DIR="$(uv tool dir)/ms-music"  # uv normalizes the name
LAUNCHER="$BIN_DIR/ms-music"
[ -x "$LAUNCHER" ] || fail "the ms-music launcher is missing ($LAUNCHER)."
ASSETS="$("$TOOL_DIR/bin/python" -c \
    'import os, ms_music.gui as g; print(os.path.join(os.path.dirname(g.__file__), "assets"))')" \
    || fail "ms_music was installed but cannot be imported."

bold "Step 3 of 3: creating the ms_music app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources" "$HOME/Desktop" \
    || fail "could not create $APP."
cp "$ASSETS/icon.icns" "$APP/Contents/Resources/icon.icns"
cat > "$APP/Contents/MacOS/ms_music" <<EOF
#!/bin/bash
# Starts the ms_music GUI in the browser (quits when the tab is closed).
# The app has no window of its own, so say that it is starting.
/usr/bin/osascript -e 'display notification "It opens in your web browser in a moment." with title "Starting ms_music"' >/dev/null 2>&1 &
exec "$LAUNCHER" "\$@"
EOF
chmod +x "$APP/Contents/MacOS/ms_music"
cat > "$APP/Contents/Info.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>ms_music</string>
    <key>CFBundleDisplayName</key><string>ms_music</string>
    <key>CFBundleIdentifier</key><string>io.github.mmattano.ms_music</string>
    <key>CFBundleExecutable</key><string>ms_music</string>
    <key>CFBundleIconFile</key><string>icon</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleVersion</key><string>1</string>
    <!-- Runs in the background; the GUI itself is a browser tab. -->
    <key>LSUIElement</key><true/>
</dict>
</plist>
EOF
touch "$APP"
ln -sfn "$APP" "$HOME/Desktop/ms_music"

echo "Preparing ms_music for its first start (about a minute)…"
# Loading everything once now lets macOS check the new libraries and
# matplotlib build its font cache here, instead of on the first double-click.
"$TOOL_DIR/bin/python" -c 'import ms_music.gui.app, ms_music.visualizations' \
    >/dev/null 2>&1 || fail "ms_music does not start (the import failed)."

VERSION="$("$TOOL_DIR/bin/python" -c 'import ms_music; print(ms_music.__version__)')"
bold "Done! ms_music $VERSION is installed."
echo "Start it any time from the ms_music icon on your Desktop or in"
echo "Applications (your home folder). It opens in your web browser."
echo "To update later, run this installer again."

if [ -z "${MS_MUSIC_SKIP_OPEN:-}" ]; then
    open "$APP"
fi
finish 0
