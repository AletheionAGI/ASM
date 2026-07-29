param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$DatasetManifest = "data\tokens_5b\manifest.json",
    [string]$DrmConfig = "configs\drm_125m_real.yaml",
    [string]$OutputRoot = "runs\probe_125m_curriculum",
    [int[]]$Seeds = @(1),
    [int64]$FastTokens = 1000000,
    [int64]$RefineTokens = 1000000,
    [int]$BatchSize = 2,
    [int]$GradAccumSteps = 8,
    [int]$SeqLen = 512,
    [double]$LearningRate = 3e-4,
    [double]$WeightDecay = 0.01,
    [double]$MaxGradNorm = 1.0,
    [string]$Precision = "bf16",
    [string]$Device = "cuda",
    [int64]$EvalTokensInterval = 1000000,
    [int64]$CheckpointTokensInterval = 1000000,
    [int]$EvalBatches = 4,
    [int]$LogInterval = 10,
    [switch]$SkipFast,
    [switch]$SkipB8Refine,
    [switch]$SkipSuperRefine,
    [switch]$DryRun,
    [switch]$DryRunForward
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path $Python)) {
    $Python = "python"
}
if (-not (Test-Path $DatasetManifest)) {
    throw "Missing dataset manifest: $DatasetManifest"
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

function Get-Summary {
    param([string]$RunDir)

    $Path = Join-Path $RunDir "summary.json"
    if (-not (Test-Path $Path)) {
        return $null
    }
    return Get-Content $Path -Raw | ConvertFrom-Json
}

function Update-Monitor {
    param(
        [string]$RunDir,
        [string]$Family,
        [int]$Seed,
        [string]$Phase
    )

    $MetricsPath = Join-Path $RunDir "metrics_latest.json"
    if (-not (Test-Path $MetricsPath)) {
        return
    }

    $MonitorPath = Join-Path $RunDir "monitor.jsonl"
    $LastTokens = 0L
    $LastElapsed = 0.0
    if (Test-Path $MonitorPath) {
        $Lines = @(Get-Content $MonitorPath)
        if ($Lines.Count -gt 0) {
            $Last = $Lines[-1] | ConvertFrom-Json
            $LastTokens = [int64]$Last.tokens_seen
            $LastElapsed = [double]$Last.elapsed_sec_cumulative
        }
    }

    $Payload = Get-Content $MetricsPath -Raw | ConvertFrom-Json
    foreach ($Row in $Payload.history) {
        $TokensSeen = [int64]$Row.tokens_seen
        if ($TokensSeen -le $LastTokens) {
            continue
        }
        $Item = [ordered]@{
            family = $Family
            phase = $Phase
            seed = $Seed
            step = [int64]$Row.step
            tokens_seen = $TokensSeen
            train_ce = $Row.train_ce
            val_ce = $Row.val_ce
            best_val_ce = $Row.best_val_ce
            tokens_per_sec = $Row.tokens_per_sec
            elapsed_sec_chunk = $Row.elapsed_sec
            elapsed_sec_cumulative = $LastElapsed + [double]$Row.elapsed_sec
        }
        Add-Content -Path $MonitorPath -Value ($Item | ConvertTo-Json -Compress)
    }
}

function Invoke-Analysis {
    & $Python scripts\analyze_time_to_quality.py --root $OutputRoot --target-margin-ce 0.01 --plateau-window 3 --min-improvement-per-million 0.003
    if ($LASTEXITCODE -ne 0) {
        throw "Analysis failed with exit code $LASTEXITCODE"
    }
}

function Invoke-DrmTrain {
    param(
        [string]$RunDir,
        [int]$Seed,
        [int64]$TargetTokens,
        [string]$Phase,
        [string]$SequenceMode,
        [int]$BlockSize,
        [int]$SuperblockSize,
        [int]$SuperblockLocalSize,
        [int]$AndersonIterations,
        [string]$CumsumStepMode,
        [string]$AndersonTransitionMode,
        [string]$ResumePath = ""
    )

    $Summary = Get-Summary -RunDir $RunDir
    if (($null -ne $Summary) -and ([int64]$Summary.tokens_seen -ge $TargetTokens)) {
        Write-Host "Reusing $Phase seed $Seed at $($Summary.tokens_seen) tokens"
        Update-Monitor -RunDir $RunDir -Family "drm" -Seed $Seed -Phase $Phase
        return
    }

    $ArgsList = @(
        "scripts\train_drm_memmap.py",
        "--config", $DrmConfig,
        "--dataset-manifest", $DatasetManifest,
        "--output-root", $RunDir,
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
        "--log-interval", "$LogInterval",
        "--sequence-mode", $SequenceMode,
        "--directional-cumsum-block-size", "$BlockSize",
        "--directional-superblock-size", "$SuperblockSize",
        "--directional-superblock-local-size", "$SuperblockLocalSize",
        "--directional-anderson-iterations", "$AndersonIterations",
        "--directional-cumsum-step-mode", $CumsumStepMode,
        "--directional-anderson-transition-mode", $AndersonTransitionMode
    )
    if ($ResumePath -ne "") {
        $ArgsList += @("--resume", $ResumePath)
    }
    if ($DryRun) {
        $ArgsList += "--dry-run"
    }
    if ($DryRunForward) {
        $ArgsList += "--dry-run-forward"
    }

    Write-Host "Running $Phase seed $Seed until $TargetTokens tokens"
    & $Python @ArgsList
    if ($LASTEXITCODE -ne 0) {
        throw "$Phase seed $Seed failed with exit code $LASTEXITCODE"
    }
    Update-Monitor -RunDir $RunDir -Family "drm" -Seed $Seed -Phase $Phase
    Invoke-Analysis
}

$TotalRefineTarget = $FastTokens + $RefineTokens

Write-Host "125M curriculum probe"
Write-Host "---------------------"
Write-Host "output_root: $OutputRoot"
Write-Host "seeds: $($Seeds -join ', ')"
Write-Host "fast_tokens: $FastTokens"
Write-Host "refine_tokens: $RefineTokens"
Write-Host "refine_target_tokens_seen: $TotalRefineTarget"
Write-Host "tokens_per_step: $($BatchSize * $SeqLen * $GradAccumSteps)"
Write-Host ""

foreach ($Seed in $Seeds) {
    $FastDir = Join-Path $OutputRoot "drm_curriculum_fast_block64_velocity_iter0_seed_$Seed"
    if (-not $SkipFast) {
        Invoke-DrmTrain `
            -RunDir $FastDir `
            -Seed $Seed `
            -TargetTokens $FastTokens `
            -Phase "fast_block64_velocity_iter0" `
            -SequenceMode "directional_block_cumsum" `
            -BlockSize 64 `
            -SuperblockSize 64 `
            -SuperblockLocalSize 8 `
            -AndersonIterations 0 `
            -CumsumStepMode "velocity" `
            -AndersonTransitionMode "velocity"
    }

    $FastCheckpoint = Join-Path $FastDir "checkpoint_last.pt"
    if (-not (Test-Path $FastCheckpoint) -and -not $DryRun) {
        throw "Missing fast checkpoint for seed $Seed`: $FastCheckpoint"
    }

    if (-not $SkipB8Refine) {
        $B8Dir = Join-Path $OutputRoot "drm_curriculum_refine_b8_iter2_from_fast_seed_$Seed"
        Invoke-DrmTrain `
            -RunDir $B8Dir `
            -Seed $Seed `
            -TargetTokens $TotalRefineTarget `
            -Phase "refine_b8_iter2_from_fast" `
            -SequenceMode "directional_block_cumsum" `
            -BlockSize 8 `
            -SuperblockSize 64 `
            -SuperblockLocalSize 8 `
            -AndersonIterations 2 `
            -CumsumStepMode "candidate" `
            -AndersonTransitionMode "candidate" `
            -ResumePath $FastCheckpoint
    }

    if (-not $SkipSuperRefine) {
        $SuperDir = Join-Path $OutputRoot "drm_curriculum_refine_super64_local8_iter2_from_fast_seed_$Seed"
        Invoke-DrmTrain `
            -RunDir $SuperDir `
            -Seed $Seed `
            -TargetTokens $TotalRefineTarget `
            -Phase "refine_super64_local8_iter2_from_fast" `
            -SequenceMode "directional_superblock_cumsum" `
            -BlockSize 8 `
            -SuperblockSize 64 `
            -SuperblockLocalSize 8 `
            -AndersonIterations 2 `
            -CumsumStepMode "velocity" `
            -AndersonTransitionMode "velocity" `
            -ResumePath $FastCheckpoint
    }
}

Invoke-Analysis

Write-Host ""
Write-Host "Done."
Write-Host "Dashboard: $(Join-Path $OutputRoot 'dashboard.html')"
Write-Host "Runs CSV:  $(Join-Path $OutputRoot 'time_to_quality_runs.csv')"
Write-Host "Agg CSV:   $(Join-Path $OutputRoot 'time_to_quality_aggregate.csv')"
