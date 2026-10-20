<#
.SYNOPSIS
Launch the E1/E2 confirmatory fits detached, so they survive a closed terminal.

.DESCRIPTION
Four jobs: E1 and E2, each under the corrected pipeline and under the registered
amendment #6(c) pipeline, per §11 sub-amendment #12, which commits to reporting
both side by side.

Two passes exist because the registered §6 bootstrap takes days and bootMer
cannot checkpoint:

  A  nsim = 0. The complete confirmatory analysis minus bootstrap percentile CIs
     and the conditional profile-CI escalation. Finishes in minutes. Its job is
     to put the registered result on disk before a multi-day run can lose it.
  B  nsim = 10000, the registered artifact. Measured at ~288 s per resample, so
     roughly 200 hours per job at 4 workers.

Pass B resamples in chunks of -ChunkSize, each saved under
data/analysis/phase_c/boot_chunks as it completes. Relaunching a job that was
killed picks up from the last completed chunk instead of starting over: run the
same command again and completed chunks are skipped. Pooling k chunks is the
same estimator as one undivided run of the same total, asserted against
confint() in src/analysis/test_bootstrap_chunked.R.

Chunk size trades checkpoint granularity against per-chunk overhead. At 400 and
8 workers per job a chunk lands roughly every 4 hours, so an interruption costs
hours rather than the days it cost on 2026-09-02.

Coefficients, Holm-adjusted p-values, Wald intervals and every directional
decision are identical across passes: they come from the fitted models, not from
the bootstrap. Only pass B output may be reported as the registered artifact.

Processes are started with Start-Process so they are owned by the OS rather than
by whatever shell launched them. Killing the terminal, or the Claude session,
does not kill them.

-Dataset picks the input and output namespace. phase_c is the 12-model set the
registered result was first computed on; phase_c15 is the 15-model roster of
amendment #13. Each reads data/analysis_inputs/<Dataset>_final and
<Dataset>_registered and writes to data/analysis/<Dataset>, so the two can never
overwrite or pool each other's artifacts or bootstrap chunks.

.EXAMPLE
  pwsh -File scripts/run_confirmatory.ps1 -Pass A
  pwsh -File scripts/run_confirmatory.ps1 -Pass A -Dataset phase_c15
  pwsh -File scripts/run_confirmatory.ps1 -Pass B -Workers 4
  pwsh -File scripts/run_confirmatory.ps1 -Pass B -Workers 4 -Only e1_corrected
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][ValidateSet('A', 'B')][string]$Pass,
    [ValidateRange(1, 64)][int]$Workers = 4,
    [ValidateRange(1, 10000)][int]$ChunkSize = 400,
    [string[]]$Only,
    [ValidateSet('phase_c', 'phase_c15')][string]$Dataset = 'phase_c'
)

$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Rscript = 'C:\Program Files\R\R-4.6.0\bin\Rscript.exe'
$OutDir = Join-Path $RepoRoot "data\analysis\$Dataset"
$LogDir = Join-Path $OutDir 'logs'
# Bootstrap chunks. Kept out of $OutDir's top level so a chunk file is never
# mistaken for a finished artifact. Namespaced per job by the R side, from the
# output filename, so e1_corrected and e1_registered cannot pool each other's
# resamples. Surviving chunks are what makes a killed run resumable.
$ChunkDir = Join-Path $OutDir 'boot_chunks'
$Manifest = Join-Path $LogDir "run_manifest_pass$Pass.json"

if (-not (Test-Path $Rscript)) { throw "Rscript not found at $Rscript" }
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path $ChunkDir | Out-Null

# nsim 0 is the crash-insurance pass; empty string lets the script use its own
# registered default of 10000 rather than a number restated here.
$Nsim = if ($Pass -eq 'A') { '0' } else { '' }
$Suffix = if ($Pass -eq 'A') { '_nobootstrap' } else { '' }

$Jobs = @(
    @{ name = 'e1_corrected';  script = 'e1_checkpoint_cascade.R';  inputs = "${Dataset}_final" }
    @{ name = 'e2_corrected';  script = 'e2_prompt_manipulation.R'; inputs = "${Dataset}_final" }
    @{ name = 'e1_registered'; script = 'e1_checkpoint_cascade.R';  inputs = "${Dataset}_registered" }
    @{ name = 'e2_registered'; script = 'e2_prompt_manipulation.R'; inputs = "${Dataset}_registered" }
)
if ($Only) { $Jobs = $Jobs | Where-Object { $Only -contains $_.name } }
if (-not $Jobs) { throw "No jobs matched -Only $($Only -join ',')" }

# Refuse to clobber a finished artifact. Re-running after a result has been seen
# is the one thing a one-shot confirmatory analysis must never do silently.
foreach ($job in $Jobs) {
    $target = Join-Path $OutDir "$($job.name)$Suffix.json"
    if (Test-Path $target) {
        throw "$target already exists. Move or delete it deliberately before re-running; a silent overwrite of a confirmatory result is not something this script will do for you."
    }
}

$records = @()
foreach ($job in $Jobs) {
    $d = "data/analysis_inputs/$($job.inputs)"
    $output = "data/analysis/$Dataset/$($job.name)$Suffix.json"
    $arguments = @(
        "src/analysis/$($job.script)"
        '--generations',   "$d/generations.jsonl"
        '--verifications', "$d/verifications.jsonl"
        '--questions',     "$d/questions.jsonl"
        '--models',        "$d/models.jsonl"
        '--output',        $output
    )
    $log = Join-Path $LogDir "$($job.name)_pass$Pass.log"
    $err = Join-Path $LogDir "$($job.name)_pass$Pass.err"

    # Set for the child only. A stray value in the parent shell silently changing
    # a confirmatory run is exactly the failure this avoids.
    $env:CITATION_BOOT_NCPUS = "$Workers"
    $env:CITATION_BOOTSTRAP_NSIM = $Nsim
    $env:CITATION_BOOT_CHUNK_DIR = $ChunkDir
    $env:CITATION_BOOT_CHUNK = "$ChunkSize"

    $proc = Start-Process -FilePath $Rscript -ArgumentList $arguments `
        -WorkingDirectory $RepoRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $log -RedirectStandardError $err

    $records += [ordered]@{
        name       = $job.name
        pid        = $proc.Id
        started    = (Get-Date).ToString('s')
        output     = $output
        log        = $log
        workers    = $(if ($Pass -eq 'A') { 0 } else { $Workers })
        nsim       = $(if ($Pass -eq 'A') { 0 } else { 10000 })
        command    = "$Rscript $($arguments -join ' ')"
    }
    Write-Host ("launched {0,-15} pid {1,-8} -> {2}" -f $job.name, $proc.Id, $output)
}

Remove-Item Env:\CITATION_BOOT_NCPUS, Env:\CITATION_BOOTSTRAP_NSIM, `
            Env:\CITATION_BOOT_CHUNK_DIR, Env:\CITATION_BOOT_CHUNK `
            -ErrorAction SilentlyContinue

@{
    pass       = $Pass
    started    = (Get-Date).ToString('s')
    workers    = $Workers
    chunk_size = $(if ($Pass -eq 'A') { 0 } else { $ChunkSize })
    chunk_dir  = $ChunkDir
    jobs       = $records
} | ConvertTo-Json -Depth 5 | Set-Content -Path $Manifest -Encoding utf8

Write-Host "`nmanifest: $Manifest"
Write-Host "status:   pwsh -File scripts/check_confirmatory.ps1 -Pass $Pass"
