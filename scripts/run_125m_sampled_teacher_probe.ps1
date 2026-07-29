param(
    [string]$OutputRoot = "runs\probe_125m_sampled_teacher",
    [int[]]$Seeds = @(1),
    [int64]$TargetTokens = 1000000,
    [int]$SeqLen = 512,
    [int]$BatchSize = 2,
    [int]$GradAccumSteps = 8,
    [int64]$EvalTokensInterval = 1000000,
    [int64]$CheckpointTokensInterval = 1000000,
    [double[]]$Weights = @(0.03, 0.05, 0.10),
    [int[]]$Intervals = @(16, 8),
    [int]$LocalSize = 8,
    [ValidateSet("candidate", "velocity")]
    [string]$TeacherMode = "candidate",
    [switch]$IncludeGpt2,
    [switch]$DryRun,
    [switch]$DryRunForward
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Runner = ".\scripts\run_125m_150m_multiseed_competition.ps1"
if (-not (Test-Path $Runner)) {
    throw "Missing runner: $Runner"
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

Write-Host "125M sampled b8-teacher probe"
Write-Host "-----------------------------"
Write-Host "output_root: $OutputRoot"
Write-Host "seeds: $($Seeds -join ', ')"
Write-Host "target_tokens: $TargetTokens"
Write-Host "weights: $($Weights -join ', ')"
Write-Host "intervals: $($Intervals -join ', ')"
Write-Host "teacher_mode: $TeacherMode"
Write-Host ""

function Invoke-Probe {
    param(
        [string]$Name,
        [double]$Weight,
        [int]$Interval,
        [switch]$SkipDrm,
        [switch]$SkipGpt2
    )

    $ArgsList = @{
        OutputRoot = Join-Path $OutputRoot $Name
        Seeds = $Seeds
        TargetTokens = $TargetTokens
        SeqLen = $SeqLen
        BatchSize = $BatchSize
        GradAccumSteps = $GradAccumSteps
        EvalTokensInterval = $EvalTokensInterval
        CheckpointTokensInterval = $CheckpointTokensInterval
        DrmSequenceMode = "directional_block_cumsum"
        DrmBlockSize = 64
        DrmAndersonIterations = 0
        DrmCumsumStepMode = "velocity"
        DrmSampledConsistencyWeight = $Weight
        DrmSampledConsistencyInterval = $Interval
        DrmSampledConsistencyLocalSize = $LocalSize
        DrmSampledConsistencyTeacherMode = $TeacherMode
    }
    if ($SkipDrm) {
        $ArgsList.SkipDrm = $true
    }
    if ($SkipGpt2) {
        $ArgsList.SkipGpt2 = $true
    }
    if ($DryRun) {
        $ArgsList.DryRun = $true
    }
    if ($DryRunForward) {
        $ArgsList.DryRunForward = $true
    }

    & $Runner @ArgsList
    if ($LASTEXITCODE -ne 0) {
        throw "Probe failed with exit code ${LASTEXITCODE}: $Name"
    }
}

Invoke-Probe -Name "baseline_block64_velocity_iter0" -Weight 0.0 -Interval 8 -SkipGpt2

foreach ($Interval in $Intervals) {
    foreach ($Weight in $Weights) {
        $WeightName = "$Weight".Replace(".", "p")
        Invoke-Probe -Name "sampled_b8_teacher_i$($Interval)_w$($WeightName)" -Weight $Weight -Interval $Interval -SkipGpt2
    }
}

if ($IncludeGpt2) {
    Invoke-Probe -Name "gpt2_reference" -Weight 0.0 -Interval 8 -SkipDrm
}

Write-Host ""
Write-Host "Done."
Write-Host "Probe root: $OutputRoot"
