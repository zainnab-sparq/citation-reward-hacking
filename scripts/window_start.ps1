<#
.SYNOPSIS
Start the 15-model confirmatory work inside Zain's overnight compute window.

.DESCRIPTION
Zain lends the machine from 1am to 5am CT, Monday to Friday. Task Scheduler
runs this at 1am and window_stop.ps1 at 5am.

What runs (all single-threaded R processes, so well under half the machine):

  * Pass A of E1/E2 under both pipelines on the phase_c15 inputs (the one-shot
    registered fit). Only jobs whose artifact is missing are launched, so a job
    killed at 5am reruns the next night and a finished one is never re-run.
  * The drift diagnostics with the robustness panel on the same inputs. The
    panel re-estimates the registered contrasts, which is why it runs alongside
    Pass A and never before it.

The bootstrap (Pass B) is deliberately absent: the plan is Wald intervals plus a
disclosed amendment #14 deviation.

PIDs are written to logs/window_pids.json so window_stop.ps1 can end exactly
these processes and nothing else.
#>
[CmdletBinding()]
param(
    [ValidateSet('phase_c15')][string]$Dataset = 'phase_c15'
)

$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Rscript = 'C:\Program Files\R\R-4.6.0\bin\Rscript.exe'
$OutDir = Join-Path $RepoRoot "data\analysis\$Dataset"
$LogDir = Join-Path $OutDir 'logs'
$PidFile = Join-Path $LogDir 'window_pids.json'
$WindowLog = Join-Path $LogDir 'window.log'
$PassAJobs = @('e1_corrected', 'e2_corrected', 'e1_registered', 'e2_registered')
$DriftOutput = "data/analysis/$Dataset/drift_interaction_withpanel.json"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Set-Location $RepoRoot

function Write-WindowLog([string]$message) {
    $line = '{0}  {1}' -f (Get-Date).ToString('s'), $message
    Add-Content -Path $WindowLog -Value $line
}

$missing = @($PassAJobs | Where-Object { -not (Test-Path (Join-Path $OutDir "${_}_nobootstrap.json")) })
$hasDrift = Test-Path (Join-Path $RepoRoot $DriftOutput)
if (-not $missing -and $hasDrift) {
    Write-WindowLog 'nothing to do: every Pass A artifact and the drift panel exist'
    exit 0
}

$launched = @()
if ($missing) {
    Write-WindowLog ("Pass A launching: {0}" -f ($missing -join ', '))
    & (Join-Path $PSScriptRoot 'run_confirmatory.ps1') -Pass A -Dataset $Dataset -Only $missing | Out-Null
    $manifest = Get-Content (Join-Path $LogDir 'run_manifest_passA.json') -Raw | ConvertFrom-Json
    $launched += @($manifest.jobs | ForEach-Object { [ordered]@{ name = $_.name; pid = $_.pid } })
}

if (-not $hasDrift) {
    $d = "data/analysis_inputs/${Dataset}_final"
    $arguments = @(
        'src/analysis/phase_c_drift_interaction.R'
        '--generations',   "$d/generations.jsonl"
        '--verifications', "$d/verifications.jsonl"
        '--questions',     "$d/questions.jsonl"
        '--models',        "$d/models.jsonl"
        '--era',           "$d/judgment_era.jsonl"
        '--output',        $DriftOutput
        '--robustness-panel'
    )
    $proc = Start-Process -FilePath $Rscript -ArgumentList $arguments `
        -WorkingDirectory $RepoRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $LogDir 'drift_withpanel.log') `
        -RedirectStandardError (Join-Path $LogDir 'drift_withpanel.err')
    $launched += [ordered]@{ name = 'drift_withpanel'; pid = $proc.Id }
    Write-WindowLog "drift + panel launching: pid $($proc.Id)"
}

$launched | ConvertTo-Json -Depth 3 | Set-Content -Path $PidFile -Encoding ascii
Write-WindowLog ("started: {0}" -f (($launched | ForEach-Object { "$($_.name)=$($_.pid)" }) -join ' '))
