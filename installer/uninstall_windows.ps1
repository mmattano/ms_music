# Remove ms_music and its shortcuts. Asks before deleting the cache of
# loaded data. (uv stays installed; it's small and shared.)
#
# Environment (for testing): MS_MUSIC_REMOVE_CACHE=yes|no answers the cache
# question without asking; MS_MUSIC_NO_PAUSE=1 skips the final key press.

$ErrorActionPreference = 'Continue'
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
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"

Write-Host ""; Write-Host "Removing ms_music" -ForegroundColor Cyan
if (Get-Command uv -ErrorAction SilentlyContinue) {
    # A running ms_music keeps its files locked:
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
    & uv tool uninstall ms_music 2>$null
    if ($LASTEXITCODE -eq 0) { Write-Host "Removed the ms_music program." }
    else { Write-Host "The ms_music program was not installed." }
}
foreach ($Folder in @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))) {
    Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $Folder 'ms_music.lnk')
}
Write-Host "Removed the shortcuts."

$Cache = if ($env:MS_MUSIC_CACHE_DIR) { $env:MS_MUSIC_CACHE_DIR } else { Join-Path $env:USERPROFILE '.cache\ms_music' }
if (Test-Path $Cache) {
    $Answer = $env:MS_MUSIC_REMOVE_CACHE
    if (-not $Answer) { $Answer = Read-Host "Also delete cached copies of loaded data in $Cache? [y/N]" }
    if ($Answer -match '^(y|yes)$') { Remove-Item -Recurse -Force $Cache; Write-Host "Deleted the cache." }
    else { Write-Host "Kept the cache." }
}
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue (Join-Path $env:LOCALAPPDATA 'ms_music')

Write-Host ""; Write-Host "ms_music has been removed." -ForegroundColor Cyan
if (-not $env:MS_MUSIC_NO_PAUSE) { Read-Host "Press Enter to close this window" | Out-Null }
