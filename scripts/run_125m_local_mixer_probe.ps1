param(
    [string]$OutputRoot = "runs\probe_125m_local_mixer",
    [int[]]$Seeds = @(1),
    [int64]$TargetTokens = 1000000,
    [int]$SeqLen = 512,
    [int]$BatchSize = 2,
    [int]$GradAccumSteps = 8,
    [int64]$EvalTokensInterval = 1000000,
    [int64]$CheckpointTokensInterval = 1000000,
    [double[]]$Scales = @(0.05, 0.10, 0.20),
    [int[]]$HiddenSizes = @(256),
    [int[]]$KernelSizes = @(8),
    [int[]]$Layers = @(1, 2),
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

Write-Host "125M local causal mixer probe"
Write-Host "----------------------------"
Write-Host "output_root: $OutputRoot"
Write-Host "seeds: $($Seeds -join ', ')"
Write-Host "target_tokens: $TargetTokens"
Write-Host "scales: $($Scales -join ', ')"
Write-Host "hidden_sizes: $($HiddenSizes -join ', ')"
Write-Host "kernel_sizes: $($KernelSizes -join ', ')"
Write-Host "layers: $($Layers -join ', ')"
Write-Host ""

function Invoke-Probe {
    param(
        [string]$Name,
        [string]$Mixer,
        [int]$HiddenSize,
        [int]$KernelSize,
        [int]$LayerCount,
        [double]$Scale,
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
        DrmLocalMixer = $Mixer
        DrmLocalMixerHiddenSize = $HiddenSize
        DrmLocalMixerKernelSize = $KernelSize
        DrmLocalMixerLayers = $LayerCount
        DrmLocalMixerScale = $Scale
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

Invoke-Probe -Name "baseline_block64_velocity_iter0" -Mixer "none" -HiddenSize 256 -KernelSize 8 -LayerCount 1 -Scale 0.0 -SkipGpt2

foreach ($HiddenSize in $HiddenSizes) {
    foreach ($KernelSize in $KernelSizes) {
        foreach ($LayerCount in $Layers) {
            foreach ($Scale in $Scales) {
                $ScaleName = "$Scale".Replace(".", "p")
                Invoke-Probe `
                    -Name "local_mixer_h$($HiddenSize)_k$($KernelSize)_l$($LayerCount)_s$($ScaleName)" `
                    -Mixer "causal_conv" `
                    -HiddenSize $HiddenSize `
                    -KernelSize $KernelSize `
                    -LayerCount $LayerCount `
                    -Scale $Scale `
                    -SkipGpt2
            }
        }
    }
}

if ($IncludeGpt2) {
    Invoke-Probe -Name "gpt2_reference" -Mixer "none" -HiddenSize 256 -KernelSize 8 -LayerCount 1 -Scale 0.0 -SkipDrm
}

Write-Host ""
Write-Host "Done."
Write-Host "Probe root: $OutputRoot"
