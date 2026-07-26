param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$DatasetManifest = "data\tokens\manifest.json",
    [string]$DrmConfig = "configs\drm_125m.yaml",
    [string]$OutputRoot = "runs\tta_multiseed_confirmation",
    [int[]]$Seeds = @(1, 2, 3),
    [int64]$InitialTargetTokens = 10000000,
    [int64]$MaxTokens = 100000000,
    [int64]$ChunkTokens = 1000000,
    [int]$BatchSize = 16,
    [int]$GradAccumSteps = 1,
    [int]$SeqLen = 64,
    [double]$LearningRate = 3e-4,
    [double]$WeightDecay = 0.01,
    [double]$MaxGradNorm = 1.0,
    [string]$Precision = "bf16",
    [string]$Device = "cuda",
    [int64]$EvalTokensInterval = 1000000,
    [int64]$CheckpointTokensInterval = 10000000,
    [int]$EvalBatches = 8,
    [int]$PlateauWindow = 3,
    [double]$MinImprovementPerMillion = 0.003,
    [double]$TargetMarginCe = 0.01,
    [int]$Gpt2LogInterval = 250,
    [switch]$AllowGpt2EarlyPlateau
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RequireGpt2AtLeastDrmTokens = -not $AllowGpt2EarlyPlateau

Write-Host "TTA multi-seed confirmation"
Write-Host "---------------------------"
Write-Host "seeds: $($Seeds -join ', ')"
Write-Host "output_root: $OutputRoot"
Write-Host "initial_target_tokens: $InitialTargetTokens"
Write-Host "max_tokens: $MaxTokens"
Write-Host "chunk_tokens: $ChunkTokens"
Write-Host "target_margin_ce: $TargetMarginCe"
Write-Host "plateau_window: $PlateauWindow"
Write-Host "min_improvement_per_million: $MinImprovementPerMillion"
Write-Host "require_gpt2_at_least_drm_tokens: $RequireGpt2AtLeastDrmTokens"
Write-Host ""

$RunArgs = @{
    Python = $Python
    DatasetManifest = $DatasetManifest
    DrmConfig = $DrmConfig
    OutputRoot = $OutputRoot
    Seeds = $Seeds
    InitialTargetTokens = $InitialTargetTokens
    MaxTokens = $MaxTokens
    ChunkTokens = $ChunkTokens
    BatchSize = $BatchSize
    GradAccumSteps = $GradAccumSteps
    SeqLen = $SeqLen
    LearningRate = $LearningRate
    WeightDecay = $WeightDecay
    MaxGradNorm = $MaxGradNorm
    Precision = $Precision
    Device = $Device
    EvalTokensInterval = $EvalTokensInterval
    CheckpointTokensInterval = $CheckpointTokensInterval
    EvalBatches = $EvalBatches
    PlateauWindow = $PlateauWindow
    MinImprovementPerMillion = $MinImprovementPerMillion
    TargetMarginCe = $TargetMarginCe
    Gpt2LogInterval = $Gpt2LogInterval
    RequireGpt2AtLeastDrmTokens = $RequireGpt2AtLeastDrmTokens
}

& .\scripts\run_time_to_quality.ps1 @RunArgs

if ($LASTEXITCODE -ne 0) {
    throw "TTA multi-seed confirmation failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Done."
Write-Host "Dashboard: $(Join-Path $OutputRoot 'dashboard.html')"
Write-Host "Status:    $(Join-Path $OutputRoot 'time_to_quality_status.json')"
Write-Host "Aggregate: $(Join-Path $OutputRoot 'time_to_quality_aggregate.csv')"
