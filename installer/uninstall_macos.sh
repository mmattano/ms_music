#!/bin/bash
# Remove ms_music, its app and Desktop shortcut. Asks before deleting the
# cache of loaded data. (uv stays installed; it's small and shared.)
#
# Environment (for testing): MS_MUSIC_REMOVE_CACHE=yes|no answers the
# cache question without asking.

set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"

finish() {
    if [[ "$0" == *.command ]] && [ -e /dev/tty ]; then
        printf '\nPress any key to close this window…'
        read -r -n 1 _ </dev/tty || true
    fi
    exit "$1"
}

printf '\n\033[1mRemoving ms_music\033[0m\n'
if command -v uv >/dev/null 2>&1; then
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
    uv tool uninstall ms_music 2>/dev/null && echo "Removed the ms_music program." \
        || echo "The ms_music program was not installed."
fi
rm -rf "$HOME/Applications/ms_music.app"
rm -f "$HOME/Desktop/ms_music"
echo "Removed the app and the Desktop shortcut."

CACHE="${MS_MUSIC_CACHE_DIR:-$HOME/.cache/ms_music}"
if [ -d "$CACHE" ]; then
    answer="${MS_MUSIC_REMOVE_CACHE:-}"
    if [ -z "$answer" ] && [ -e /dev/tty ]; then
        printf 'Also delete cached copies of loaded data in %s? [y/N] ' "$CACHE"
        read -r answer </dev/tty || answer=""
    fi
    case "$answer" in
        y|Y|yes|YES) rm -rf "$CACHE"; echo "Deleted the cache." ;;
        *) echo "Kept the cache." ;;
    esac
fi
rm -rf "$HOME/Library/Logs/ms_music"

printf '\n\033[1mms_music has been removed.\033[0m\n'
finish 0
