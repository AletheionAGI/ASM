param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$DatasetManifest = "data\tokens\manifest.json",
    [string]$DrmConfig = "configs\drm_125m.yaml",
    [string]$OutputRoot = "runs\compare_10m_seed1",
    [int64]$TargetTokens = 10000000,
    [int]$Seed = 1,
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
    [double]$LargeAdvantageCe = 0.20,
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path $Python)) {
    $Python = "python"
}

$DrmRun = Join-Path $OutputRoot "drm_37m_causal_anderson_b8_seed_$Seed"
$Gpt2Run = Join-Path $OutputRoot "gpt2_36m_seed_$Seed"

function Invoke-RunIfNeeded {
    param(
        [string]$RunDir,
        [string[]]$ArgsList
    )

    $SummaryPath = Join-Path $RunDir "summary.json"
    if ((Test-Path $SummaryPath) -and (-not $Force)) {
        Write-Host "Reusing existing run: $RunDir"
        return
    }

    Write-Host "Running: $RunDir"
    & $Python @ArgsList
    if ($LASTEXITCODE -ne 0) {
        throw "Run failed with exit code $LASTEXITCODE`: $RunDir"
    }
}

function Read-RunResult {
    param([string]$RunDir)

    $SummaryPath = Join-Path $RunDir "summary.json"
    $MetricsPath = Join-Path $RunDir "metrics_latest.json"
    $ConfigPath = Join-Path $RunDir "run_config.json"

    if (-not (Test-Path $SummaryPath)) {
        throw "Missing summary: $SummaryPath"
    }
    if (-not (Test-Path $MetricsPath)) {
        throw "Missing metrics: $MetricsPath"
    }
    if (-not (Test-Path $ConfigPath)) {
        throw "Missing run config: $ConfigPath"
    }

    $Summary = Get-Content $SummaryPath -Raw | ConvertFrom-Json
    $Metrics = Get-Content $MetricsPath -Raw | ConvertFrom-Json
    $RunConfig = Get-Content $ConfigPath -Raw | ConvertFrom-Json

    [pscustomobject]@{
        RunDir = $RunDir
        ParameterCount = [int64]$Summary.parameter_count
        BestValCe = [double]$Summary.best_val_ce
        TokensSeen = [int64]$Summary.tokens_seen
        TokensPerStep = [int64]$RunConfig.tokens_per_step
        TokensPerSec = [double]$Metrics.latest.tokens_per_sec
        ElapsedSec = [double]$Metrics.latest.elapsed_sec
    }
}

$DrmArgs = @(
    "scripts\train_drm_memmap.py",
    "--config", $DrmConfig,
    "--dataset-manifest", $DatasetManifest,
    "--output-root", $DrmRun,
    "--target-tokens", "$TargetTokens",
    "--batch-size", "$BatchSize",
    "--grad-accum-steps", "$GradAccumSteps",
    "--seq-len", "$SeqLen",
    "--lr", "$LearningRate",
    "--weight-decay", "$WeightDecay",
    "--max-grad-norm", "$MaxGradNorm",
    "--precision", $Precision,
    "--device", $Device,
    "--seed", "$Seed",
    "--eval-tokens-interval", "$EvalTokensInterval",
    "--checkpoint-tokens-interval", "$CheckpointTokensInterval",
    "--eval-batches", "$EvalBatches",
    "--log-interval", "999999",
    "--sequence-mode", "directional_block_cumsum",
    "--directional-candidate-temperature", "1.0",
    "--directional-candidate-scale", "0.01",
    "--directional-cumsum-block-size", "8",
    "--directional-anderson-iterations", "2",
    "--directional-anderson-history-size", "4",
    "--directional-anderson-ridge", "0.0001",
    "--directional-anderson-relaxation", "1.0"
)

$Gpt2Args = @(
    "scripts\train_gpt2_memmap.py",
    "--model-size", "gpt2_125m",
    "--dataset-manifest", $DatasetManifest,
    "--output-root", $Gpt2Run,
    "--target-tokens", "$TargetTokens",
    "--batch-size", "$BatchSize",
    "--grad-accum-steps", "$GradAccumSteps",
    "--seq-len", "$SeqLen",
    "--lr", "$LearningRate",
    "--weight-decay", "$WeightDecay",
    "--max-grad-norm", "$MaxGradNorm",
    "--dropout", "0.0",
    "--precision", $Precision,
    "--device", $Device,
    "--seed", "$Seed",
    "--eval-tokens-interval", "$EvalTokensInterval",
    "--checkpoint-tokens-interval", "$CheckpointTokensInterval",
    "--eval-batches", "$EvalBatches",
    "--log-interval", "250"
)

Invoke-RunIfNeeded -RunDir $DrmRun -ArgsList $DrmArgs
Invoke-RunIfNeeded -RunDir $Gpt2Run -ArgsList $Gpt2Args

$Drm = Read-RunResult -RunDir $DrmRun
$Gpt2 = Read-RunResult -RunDir $Gpt2Run
$CeAdvantage = $Gpt2.BestValCe - $Drm.BestValCe
$Gpt2Speedup = $Gpt2.TokensPerSec / [math]::Max($Drm.TokensPerSec, 1e-8)

Write-Host ""
Write-Host "10M seed $Seed comparison"
Write-Host "------------------------"
Write-Host ("DRM causal Anderson b8: params={0:N0} best_val_ce={1:N4} tokens_seen={2:N0} tokens/s={3:N0} elapsed={4:N1}s" -f $Drm.ParameterCount, $Drm.BestValCe, $Drm.TokensSeen, $Drm.TokensPerSec, $Drm.ElapsedSec)
Write-Host ("GPT-2 36M:              params={0:N0} best_val_ce={1:N4} tokens_seen={2:N0} tokens/s={3:N0} elapsed={4:N1}s" -f $Gpt2.ParameterCount, $Gpt2.BestValCe, $Gpt2.TokensSeen, $Gpt2.TokensPerSec, $Gpt2.ElapsedSec)
Write-Host ("CE advantage DRM over GPT-2: {0:N4}" -f $CeAdvantage)
Write-Host ("GPT-2 throughput multiple:   {0:N2}x" -f $Gpt2Speedup)

if ($CeAdvantage -ge $LargeAdvantageCe) {
    Write-Host "Decision: advantage is still large. Complete seeds 2 and 3, then test b16."
} else {
    Write-Host "Decision: advantage is not large enough by threshold. Inspect before spending on seeds 2 and 3."
}

