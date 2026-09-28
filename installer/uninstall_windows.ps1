# Remove ms_music and its shortcuts. Asks before deleting the cache of
# loaded data. (uv stays installed; it's small and shared.)
#
# Environment (for testing): MS_MUSIC_REMOVE_CACHE=yes|no answers the cache
# question without asking; MS_MUSIC_NO_PAUSE=1 skips the final key press.

$ErrorActionPreference = 'Continue'
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"

Write-Host ""; Write-Host "Removing ms_music" -ForegroundColor Cyan
if (Get-Command uv -ErrorAction SilentlyContinue) {
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
