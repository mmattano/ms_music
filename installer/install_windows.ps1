# Install or update ms_music for the current user (no admin rights needed).
#
# What it does:
#   1. installs "uv" (a small tool that manages Python) if it isn't there;
#   2. installs ms_music, with its own Python, in an isolated place;
#   3. creates "ms_music" shortcuts on the Desktop and in the Start menu.
# Running it again updates ms_music.
#
# Environment (for testing): MS_MUSIC_PACKAGE (install from a wheel/path
# instead of PyPI), MS_MUSIC_SKIP_OPEN=1 (don't start ms_music at the end),
# MS_MUSIC_NO_PAUSE=1 (don't wait for a key at the end).

$ErrorActionPreference = 'Stop'
# Started from PowerShell 7 (pwsh), Windows PowerShell inherits pwsh's
# module path and then fails to load its own built-in modules
# (Get-ExecutionPolicy, Get-Process, ...). Use Windows PowerShell's own.
if ($PSVersionTable.PSVersion.Major -le 5) {
    $env:PSModulePath = @(
        (Join-Path ([Environment]::GetFolderPath('MyDocuments')) 'WindowsPowerShell\Modules'),
        (Join-Path $env:ProgramFiles 'WindowsPowerShell\Modules'),
        [Environment]::GetEnvironmentVariable('PSModulePath', 'Machine')
    ) -join ';'
}
$Package = if ($env:MS_MUSIC_PACKAGE) { $env:MS_MUSIC_PACKAGE } else { 'ms_music' }
$PythonVersion = '3.12'

function Say($text) { Write-Host ""; Write-Host $text -ForegroundColor Cyan }
function Finish($code) {
    if (-not $env:MS_MUSIC_NO_PAUSE) {
        Write-Host ""; Read-Host "Press Enter to close this window" | Out-Null
    }
    exit $code
}

try {
    Say "Installing ms_music"
    Write-Host "This takes a few minutes the first time (it downloads Python and"
    Write-Host "the scientific libraries). Nothing is installed system-wide."

    Say "Step 1 of 3: getting the installer tool (uv)"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        Write-Host "uv is already installed."
    } else {
        $env:UV_NO_MODIFY_PATH = '1'
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
        if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
            throw "uv was not found after installing it."
        }
    }

    Say "Step 2 of 3: installing ms_music"
    # A running ms_music keeps its files locked, so the update would fail:
    # ask it to quit (ports as in ms_music.gui.PORT_RANGE), then make sure.
    $Stopped = $false
    foreach ($Port in 8765..8774) {
        try {
            Invoke-RestMethod -Method Post -TimeoutSec 2 -Headers @{ 'X-MS-Music' = 'quit' } "http://127.0.0.1:$Port/ms_music/quit" | Out-Null
            $Stopped = $true
        } catch { }
    }
    $ToolRoot = Join-Path (& uv tool dir).Trim() 'ms-music'
    $Running = Get-Process -ErrorAction SilentlyContinue | Where-Object {
        $_.Path -and ($_.Path.StartsWith($ToolRoot, 'OrdinalIgnoreCase') -or $_.Path -like '*\.local\bin\ms-music.exe')
    }
    if ($Running) { $Running | Stop-Process -Force -ErrorAction SilentlyContinue; $Stopped = $true }
    if ($Stopped) { Write-Host "Stopped the running ms_music."; Start-Sleep -Seconds 2 }
    # --compile-bytecode: prepare Python files now, so the first start is quick.
    & uv tool install --python $PythonVersion --upgrade --force --compile-bytecode $Package
    if ($LASTEXITCODE -ne 0) { throw "uv could not install ms_music." }

    $BinDir = (& uv tool dir --bin).Trim()
    $ToolDir = Join-Path (& uv tool dir).Trim() 'ms-music'  # uv normalizes the name
    $Launcher = Join-Path $BinDir 'ms-music.exe'
    if (-not (Test-Path $Launcher)) { throw "The ms-music launcher is missing ($Launcher)." }
    $Python = Join-Path $ToolDir 'Scripts\python.exe'
    $Assets = (& $Python -c "import os, ms_music.gui as g; print(os.path.join(os.path.dirname(g.__file__), 'assets'))").Trim()
    if ($LASTEXITCODE -ne 0) { throw "ms_music was installed but cannot be imported." }

    Say "Step 3 of 3: creating shortcuts"
    $Shell = New-Object -ComObject WScript.Shell
    $Folders = @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))
    foreach ($Folder in $Folders) {
        $Link = $Shell.CreateShortcut((Join-Path $Folder 'ms_music.lnk'))
        $Link.TargetPath = $Launcher
        $Link.IconLocation = (Join-Path $Assets 'icon.ico')
        $Link.WorkingDirectory = [Environment]::GetFolderPath('MyDocuments')
        $Link.Description = 'ms_music: mass spectrometry sonification'
        $Link.Save()
    }

    Write-Host "Preparing ms_music for its first start (about a minute)..."
    # Loading everything once now does the first-time work (virus scan of the
    # new libraries, matplotlib font cache) here, not on the first launch.
    # Windows PowerShell turns any stderr line of a redirected command into
    # an error under 'Stop' (matplotlib warns while building its font cache),
    # so judge this step by the exit code only.
    $ErrorActionPreference = 'Continue'
    & $Python -c "import ms_music.gui.app, ms_music.visualizations" 2>$null | Out-Null
    $ImportExit = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($ImportExit -ne 0) { throw "ms_music does not start (the import failed)." }

    $Version = (& $Python -c "import ms_music; print(ms_music.__version__)").Trim()
    Say "Done! ms_music $Version is installed."
    Write-Host "Start it any time from the ms_music icon on your Desktop or in the"
    Write-Host "Start menu. It opens in your web browser."
    Write-Host "To update later, run this installer again."

    if (-not $env:MS_MUSIC_SKIP_OPEN) { Start-Process $Launcher }
    Finish 0
} catch {
    Write-Host ""
    Write-Host "Sorry, something went wrong: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Please send a screenshot of this window to the ms_music maintainers."
    Finish 1
}
