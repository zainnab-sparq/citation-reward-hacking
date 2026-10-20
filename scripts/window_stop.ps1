<#
.SYNOPSIS
End the overnight compute window: stop whatever window_start.ps1 launched.

.DESCRIPTION
Runs at 5am CT. Only the PIDs recorded in logs/window_pids.json are touched,
and only if they are still Rscript processes, so a recycled PID belonging to
something else is never killed. taskkill /T takes the whole process tree.

A Pass A job stopped here leaves no artifact; window_start.ps1 relaunches it the
next weeknight. Nothing partial is ever written, because each R script writes its
JSON only after every fit finishes.
#>
[CmdletBinding()]
param(
    [ValidateSet('phase_c15')][string]$Dataset = 'phase_c15'
)

$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $RepoRoot "data\analysis\$Dataset\logs"
$PidFile = Join-Path $LogDir 'window_pids.json'
$WindowLog = Join-Path $LogDir 'window.log'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-WindowLog([string]$message) {
    $line = '{0}  {1}' -f (Get-Date).ToString('s'), $message
    Add-Content -Path $WindowLog -Value $line
}

if (-not (Test-Path $PidFile)) {
    Write-WindowLog 'stop: no pid file, nothing to stop'
    exit 0
}

$records = @(Get-Content $PidFile -Raw | ConvertFrom-Json)
foreach ($record in $records) {
    $proc = Get-Process -Id $record.pid -ErrorAction SilentlyContinue
    if (-not $proc) {
        Write-WindowLog "stop: $($record.name) pid $($record.pid) already finished"
        continue
    }
    if ($proc.ProcessName -ne 'Rscript') {
        Write-WindowLog "stop: pid $($record.pid) is now $($proc.ProcessName), not ours; left alone"
        continue
    }
    & taskkill.exe /PID $record.pid /T /F | Out-Null
    Write-WindowLog "stop: killed $($record.name) pid $($record.pid) at the window end"
}
Remove-Item $PidFile
