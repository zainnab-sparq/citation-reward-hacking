<#
.SYNOPSIS
Status of a confirmatory run launched by run_confirmatory.ps1.

.DESCRIPTION
Reads the pass manifest and reports, per job, whether the process is still
alive, whether its output landed, and the last line of its log. Distinguishes
the three states that matter on a multi-day run: still working, finished, and
died without producing anything.

A job that shows DIED needs relaunching on its own. Because the launcher refuses
to overwrite an existing artifact, restarting one job cannot disturb the three
that are still running:

  pwsh -File scripts/run_confirmatory.ps1 -Pass B -Workers 4 -Only <job name>

.EXAMPLE
  pwsh -File scripts/check_confirmatory.ps1 -Pass B
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][ValidateSet('A', 'B')][string]$Pass
)

$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Manifest = Join-Path $RepoRoot "data\analysis\phase_c\logs\run_manifest_pass$Pass.json"
if (-not (Test-Path $Manifest)) { throw "No manifest for pass ${Pass}: $Manifest" }

$run = Get-Content $Manifest -Raw | ConvertFrom-Json
$started = [datetime]$run.started
$elapsed = (Get-Date) - $started

Write-Host ("pass {0}   started {1}   elapsed {2:0.0} h   {3} worker(s)/job" -f `
    $run.pass, $run.started, $elapsed.TotalHours, $run.workers)
Write-Host ''

$alive = 0
$done = 0
foreach ($job in $run.jobs) {
    $proc = Get-Process -Id $job.pid -ErrorAction SilentlyContinue
    # A recycled PID would misreport a dead job as alive, so require that the
    # process actually be the Rscript we started, not merely that the id exists.
    $running = $proc -and $proc.ProcessName -eq 'Rscript' -and $proc.StartTime -lt (Get-Date)
    $produced = Test-Path (Join-Path $RepoRoot $job.output)

    $state = if ($produced) { 'DONE' } elseif ($running) { 'RUNNING' } else { 'DIED' }
    if ($state -eq 'RUNNING') { $alive++ }
    if ($state -eq 'DONE') { $done++ }

    $tail = ''
    if (Test-Path $job.log) {
        $tail = (Get-Content $job.log -Tail 1 -ErrorAction SilentlyContinue) -join ''
        if ($tail.Length -gt 70) { $tail = $tail.Substring(0, 70) + '...' }
    }
    Write-Host ("{0,-15} {1,-8} pid {2,-8} {3}" -f $job.name, $state, $job.pid, $tail)

    # Completed chunks are measured progress, not a projection. A job showing
    # DIED with chunks on disk resumes from them; only the current chunk is lost.
    if ($run.chunk_size -gt 0 -and $run.chunk_dir) {
        $done_chunks = @(Get-ChildItem -Path $run.chunk_dir -Filter "$($job.name)_chunk*.rds" `
                             -ErrorAction SilentlyContinue).Count
        $total_chunks = [math]::Ceiling($job.nsim / $run.chunk_size)
        if ($total_chunks -gt 0) {
            Write-Host ("                 chunks {0}/{1} ({2:0.0}% of resamples banked)" -f `
                $done_chunks, $total_chunks, (100 * $done_chunks / $total_chunks))
        }
    }

    if ($state -eq 'DIED') {
        $errFile = $job.log -replace '\.log$', '.err'
        if ((Test-Path $errFile) -and (Get-Item $errFile).Length -gt 0) {
            Write-Host ("                 stderr: " + ((Get-Content $errFile -Tail 2) -join ' | '))
        }
    }
}

$total = @($run.jobs).Count
Write-Host ''
Write-Host ("$done/$total finished, $alive still running")

if ($run.pass -eq 'B' -and $alive -gt 0) {
    # Project from chunks actually banked, not from a per-resample constant.
    # The 288 s/resample figure was measured at 4 workers per job; at 8 it is
    # ~441, because doubling workers bought only ~1.31x throughput (memory
    # bandwidth, not cores, is the limit on repeated GLMM refits). A constant
    # calibrated at one worker count silently misreports at another, so the
    # only trustworthy projection is the one the run itself produces.
    Write-Host ''
    foreach ($job in $run.jobs) {
        if (-not ($run.chunk_size -gt 0 -and $run.chunk_dir)) { continue }
        $done_chunks = @(Get-ChildItem -Path $run.chunk_dir -Filter "$($job.name)_chunk*.rds" `
                             -ErrorAction SilentlyContinue).Count
        $total_chunks = [math]::Ceiling($job.nsim / $run.chunk_size)
        # Per-chunk cost comes from the durations the run itself reported, not
        # from elapsed/done: elapsed also covers the chunk currently in flight,
        # which would inflate the estimate badly while few chunks are banked.
        $durations = @()
        if (Test-Path $job.log) {
            $durations = Get-Content $job.log -ErrorAction SilentlyContinue |
                Select-String -Pattern 'chunk \d+/\d+ done \(.*?([\d.]+) min\)' |
                ForEach-Object { [double]$_.Matches[0].Groups[1].Value / 60.0 }
        }
        if ($durations.Count -ge 1) {
            $per_chunk = ($durations | Measure-Object -Average).Average
            $remaining = ($total_chunks - $done_chunks) * $per_chunk
            Write-Host ("{0,-15} {1:0.0} h/chunk measured -> ~{2:0} h left, done ~{3}" -f `
                $job.name, $per_chunk, $remaining, (Get-Date).AddHours($remaining).ToString('MMM dd HH:mm'))
        } else {
            Write-Host ("{0,-15} no chunk finished yet after {1:0.0} h; cannot project" -f `
                $job.name, $elapsed.TotalHours)
        }
    }
}

$workers = @(Get-Process Rscript -ErrorAction SilentlyContinue).Count
Write-Host "Rscript processes on the box right now: $workers"
