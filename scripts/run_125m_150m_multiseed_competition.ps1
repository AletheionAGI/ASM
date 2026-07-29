param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$DatasetManifest = "data\tokens_5b\manifest.json",
    [string]$DrmConfig = "configs\drm_125m_real.yaml",
    [string]$OutputRoot = "runs\competition_125m_150m_multiseed",
    [int[]]$Seeds = @(1, 2, 3),
    [int64]$TargetTokens = 150000000,
    [int]$BatchSize = 2,
    [int]$GradAccumSteps = 8,
    [int]$SeqLen = 512,
    [double]$LearningRate = 3e-4,
    [double]$WeightDecay = 0.01,
    [double]$MaxGradNorm = 1.0,
    [string]$Precision = "bf16",
    [string]$Device = "cuda",
    [int64]$EvalTokensInterval = 10000000,
    [int64]$CheckpointTokensInterval = 50000000,
    [int]$EvalBatches = 4,
    [int]$DrmLogInterval = 10,
    [int]$Gpt2LogInterval = 100,
    [ValidateSet("directional_block_cumsum", "directional_superblock_cumsum")]
    [string]$DrmSequenceMode = "directional_block_cumsum",
    [int]$DrmBlockSize = 8,
    [int]$DrmSuperblockSize = 64,
    [int]$DrmSuperblockLocalSize = 8,
    [int]$DrmAndersonIterations = 2,
    [int]$DrmAndersonHistorySize = 4,
    [double]$DrmAndersonRidge = 0.0001,
    [double]$DrmAndersonRelaxation = 1.0,
    [ValidateSet("candidate", "velocity")]
    [string]$DrmAndersonTransitionMode = "candidate",
    [int]$DrmAndersonBlockStride = 1,
    [ValidateSet("trajectory", "endpoint")]
    [string]$DrmAndersonScope = "trajectory",
    [double]$DrmCandidateScale = 0.01,
    [ValidateSet("candidate", "velocity")]
    [string]$DrmCumsumStepMode = "candidate",
    [int]$DrmInnerBlockSize = 0,
    [ValidateSet("none", "causal_conv")]
    [string]$DrmLocalMixer = "none",
    [int]$DrmLocalMixerHiddenSize = 256,
    [int]$DrmLocalMixerKernelSize = 8,
    [int]$DrmLocalMixerLayers = 1,
    [double]$DrmLocalMixerScale = 0.1,
    [double]$DrmSampledConsistencyWeight = 0.0,
    [int]$DrmSampledConsistencyInterval = 8,
    [int]$DrmSampledConsistencyLocalSize = 8,
    [ValidateSet("candidate", "velocity")]
    [string]$DrmSampledConsistencyTeacherMode = "candidate",
    [double]$TargetMarginCe = 0.01,
    [int]$PlateauWindow = 3,
    [double]$MinImprovementPerMillion = 0.003,
    [switch]$SkipDrm,
    [switch]$SkipGpt2,
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

function Format-Duration {
    param([double]$Seconds)

    if ($Seconds -lt 3600) {
        return ("{0:N1} min" -f ($Seconds / 60.0))
    }
    if ($Seconds -lt 86400) {
        return ("{0:N1} h" -f ($Seconds / 3600.0))
    }
    return ("{0:N1} days" -f ($Seconds / 86400.0))
}

function Get-Summary {
    param([string]$RunDir)

    $Path = Join-Path $RunDir "summary.json"
    if (-not (Test-Path $Path)) {
        return $null
    }
    return Get-Content $Path -Raw | ConvertFrom-Json
}

function Get-CheckpointResumeValue {
    param([string]$RunDir)

    $LastCheckpoint = Join-Path $RunDir "checkpoint_last.pt"
    if (Test-Path $LastCheckpoint) {
        return $LastCheckpoint
    }
    $LatestCheckpoint = Join-Path $RunDir "checkpoint_latest.pt"
    if (Test-Path $LatestCheckpoint) {
        return "latest"
    }
    return ""
}

function Update-Monitor {
    param(
        [string]$RunDir,
        [string]$Family,
        [int]$Seed
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
    $ArgsList = @(
        "scripts\analyze_time_to_quality.py",
        "--root", $OutputRoot,
        "--target-margin-ce", "$TargetMarginCe",
        "--plateau-window", "$PlateauWindow",
        "--min-improvement-per-million", "$MinImprovementPerMillion"
    )
    & $Python @ArgsList
    if ($LASTEXITCODE -ne 0) {
        throw "Analysis failed with exit code $LASTEXITCODE"
    }
}

function Invoke-DrmRun {
    param([int]$Seed)

    $ModeName = if ($DrmSequenceMode -eq "directional_superblock_cumsum") {
        "super$($DrmSuperblockSize)_local$($DrmSuperblockLocalSize)"
    } else {
        "b$($DrmBlockSize)"
    }
    $SampledName = if ($DrmSampledConsistencyWeight -gt 0.0) {
        "_sampled$($DrmSampledConsistencyInterval)x$($DrmSampledConsistencyLocalSize)_w$($DrmSampledConsistencyWeight)"
    } else {
        ""
    }
    $MixerName = if ($DrmLocalMixer -ne "none") {
        "_mixh$($DrmLocalMixerHiddenSize)_k$($DrmLocalMixerKernelSize)_l$($DrmLocalMixerLayers)_scale$($DrmLocalMixerScale)"
    } else {
        ""
    }
    $RunName = "drm_125m_real_causal_anderson_$($ModeName)_step$($DrmCumsumStepMode)_and$($DrmAndersonTransitionMode)_$($DrmAndersonScope)_s$($DrmAndersonBlockStride)$($MixerName)$($SampledName)_seed_$Seed"
    $RunDir = Join-Path $OutputRoot $RunName
    $Summary = Get-Summary -RunDir $RunDir
    if (($null -ne $Summary) -and ([int64]$Summary.tokens_seen -ge $TargetTokens)) {
        Write-Host "Reusing completed DRM seed $Seed at $($Summary.tokens_seen) tokens"
        Update-Monitor -RunDir $RunDir -Family "drm" -Seed $Seed
        return
    }

    $Resume = Get-CheckpointResumeValue -RunDir $RunDir
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
        "--log-interval", "$DrmLogInterval",
        "--sequence-mode", "$DrmSequenceMode",
        "--directional-candidate-temperature", "1.0",
        "--directional-candidate-scale", "$DrmCandidateScale",
        "--directional-cumsum-step-mode", "$DrmCumsumStepMode",
        "--directional-cumsum-block-size", "$DrmBlockSize",
        "--directional-superblock-size", "$DrmSuperblockSize",
        "--directional-superblock-local-size", "$DrmSuperblockLocalSize",
        "--directional-cumsum-inner-block-size", "$DrmInnerBlockSize",
        "--directional-anderson-iterations", "$DrmAndersonIterations",
        "--directional-anderson-history-size", "$DrmAndersonHistorySize",
        "--directional-anderson-ridge", "$DrmAndersonRidge",
        "--directional-anderson-relaxation", "$DrmAndersonRelaxation",
        "--directional-anderson-transition-mode", "$DrmAndersonTransitionMode",
        "--directional-anderson-block-stride", "$DrmAndersonBlockStride",
        "--directional-anderson-scope", "$DrmAndersonScope",
        "--directional-local-mixer", "$DrmLocalMixer",
        "--directional-local-mixer-hidden-size", "$DrmLocalMixerHiddenSize",
        "--directional-local-mixer-kernel-size", "$DrmLocalMixerKernelSize",
        "--directional-local-mixer-layers", "$DrmLocalMixerLayers",
        "--directional-local-mixer-scale", "$DrmLocalMixerScale",
        "--lambda-sampled-block-consistency", "$DrmSampledConsistencyWeight",
        "--sampled-block-consistency-interval", "$DrmSampledConsistencyInterval",
        "--sampled-block-consistency-local-size", "$DrmSampledConsistencyLocalSize",
        "--sampled-block-consistency-teacher-mode", "$DrmSampledConsistencyTeacherMode"
    )
    if ($Resume -ne "") {
        $ArgsList += @("--resume", $Resume)
    }
    if ($DryRun) {
        $ArgsList += "--dry-run"
    }
    if ($DryRunForward) {
        $ArgsList += "--dry-run-forward"
    }

    Write-Host "Running DRM 125M causal Anderson $ModeName seed $Seed until $TargetTokens tokens"
    & $Python @ArgsList
    if ($LASTEXITCODE -ne 0) {
        throw "DRM seed $Seed failed with exit code $LASTEXITCODE"
    }
    Update-Monitor -RunDir $RunDir -Family "drm" -Seed $Seed
    Invoke-Analysis
}

function Invoke-Gpt2Run {
    param([int]$Seed)

    $RunName = "gpt2_125m_real_seed_$Seed"
    $RunDir = Join-Path $OutputRoot $RunName
    $Summary = Get-Summary -RunDir $RunDir
    if (($null -ne $Summary) -and ([int64]$Summary.tokens_seen -ge $TargetTokens)) {
        Write-Host "Reusing completed GPT-2 seed $Seed at $($Summary.tokens_seen) tokens"
        Update-Monitor -RunDir $RunDir -Family "gpt2" -Seed $Seed
        return
    }

    $Resume = Get-CheckpointResumeValue -RunDir $RunDir
    $ArgsList = @(
        "scripts\train_gpt2_memmap.py",
        "--model-size", "gpt2_125m_real",
        "--dataset-manifest", $DatasetManifest,
        "--output-root", $RunDir,
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
        "--log-interval", "$Gpt2LogInterval"
    )
    if ($Resume -ne "") {
        $ArgsList += @("--resume", $Resume)
    }
    if ($DryRun) {
        $ArgsList += "--dry-run"
    }
    if ($DryRunForward) {
        $ArgsList += "--dry-run-forward"
    }

    Write-Host "Running GPT-2 125M real seed $Seed until $TargetTokens tokens"
    & $Python @ArgsList
    if ($LASTEXITCODE -ne 0) {
        throw "GPT-2 seed $Seed failed with exit code $LASTEXITCODE"
    }
    Update-Monitor -RunDir $RunDir -Family "gpt2" -Seed $Seed
    Invoke-Analysis
}

$TokensPerStep = $BatchSize * $SeqLen * $GradAccumSteps
$StepsPerRun = [math]::Ceiling($TargetTokens / [double]$TokensPerStep)
$PriorDrmTps = 848.0
$PriorGpt2Tps = 26179.0
$PriorTotalSeconds = 0.0
if (-not $SkipDrm) {
    $PriorTotalSeconds += ($TargetTokens * $Seeds.Count) / $PriorDrmTps
}
if (-not $SkipGpt2) {
    $PriorTotalSeconds += ($TargetTokens * $Seeds.Count) / $PriorGpt2Tps
}

Write-Host "125M / 150M-token multiseed competition"
Write-Host "--------------------------------------"
Write-Host "output_root: $OutputRoot"
Write-Host "dataset: $DatasetManifest"
Write-Host "seeds: $($Seeds -join ', ')"
Write-Host "target_tokens_per_run: $TargetTokens"
Write-Host "tokens_per_step: $TokensPerStep"
Write-Host "steps_per_run: $StepsPerRun"
Write-Host "eval_every_tokens: $EvalTokensInterval"
Write-Host "checkpoint_every_tokens: $CheckpointTokensInterval"
Write-Host "drm_sequence_mode: $DrmSequenceMode"
Write-Host "drm_block_size: $DrmBlockSize"
Write-Host "drm_superblock_size: $DrmSuperblockSize"
Write-Host "drm_superblock_local_size: $DrmSuperblockLocalSize"
Write-Host "drm_anderson_iterations: $DrmAndersonIterations"
Write-Host "drm_anderson_history_size: $DrmAndersonHistorySize"
Write-Host "drm_anderson_transition_mode: $DrmAndersonTransitionMode"
Write-Host "drm_anderson_scope: $DrmAndersonScope"
Write-Host "drm_anderson_block_stride: $DrmAndersonBlockStride"
Write-Host "drm_cumsum_step_mode: $DrmCumsumStepMode"
Write-Host "drm_local_mixer: $DrmLocalMixer"
Write-Host "drm_local_mixer_hidden_size: $DrmLocalMixerHiddenSize"
Write-Host "drm_local_mixer_kernel_size: $DrmLocalMixerKernelSize"
Write-Host "drm_local_mixer_layers: $DrmLocalMixerLayers"
Write-Host "drm_local_mixer_scale: $DrmLocalMixerScale"
Write-Host "drm_sampled_consistency_weight: $DrmSampledConsistencyWeight"
Write-Host "drm_sampled_consistency_interval: $DrmSampledConsistencyInterval"
Write-Host "drm_sampled_consistency_local_size: $DrmSampledConsistencyLocalSize"
Write-Host "drm_sampled_consistency_teacher_mode: $DrmSampledConsistencyTeacherMode"
Write-Host "prior_125m_estimate_total: $(Format-Duration $PriorTotalSeconds) based on ~848 DRM tok/s and ~26,179 GPT-2 tok/s"
Write-Host ""

if (-not $SkipDrm) {
    foreach ($Seed in $Seeds) {
        Invoke-DrmRun -Seed $Seed
    }
}

if (-not $SkipGpt2) {
    foreach ($Seed in $Seeds) {
        Invoke-Gpt2Run -Seed $Seed
    }
}

Invoke-Analysis

Write-Host ""
Write-Host "Done."
Write-Host "Dashboard: $(Join-Path $OutputRoot 'dashboard.html')"
Write-Host "Status:    $(Join-Path $OutputRoot 'time_to_quality_status.json')"
Write-Host "Runs CSV:  $(Join-Path $OutputRoot 'time_to_quality_runs.csv')"
Write-Host "Agg CSV:   $(Join-Path $OutputRoot 'time_to_quality_aggregate.csv')"
