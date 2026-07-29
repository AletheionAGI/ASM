param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$DatasetManifest = "data\tokens_5b\manifest.json",
    [string]$OutputRoot = "runs\competition_125m_local_mixer_h256_l2_s02_10m",
    [int[]]$ConfirmSeeds = @(1, 2, 3),
    [int64]$ConfirmTargetTokens = 10000000,
    [string]$LongOutputRoot = "runs\competition_125m_local_mixer_h256_l2_s02_150m",
    [int[]]$LongSeeds = @(1, 2, 3),
    [int64]$LongTargetTokens = 150000000,
    [int]$SeqLen = 512,
    [int]$BatchSize = 2,
    [int]$GradAccumSteps = 8,
    [int64]$EvalTokensInterval = 1000000,
    [int64]$ConfirmCheckpointTokensInterval = 10000000,
    [int64]$LongCheckpointTokensInterval = 50000000,
    [switch]$SkipConfirm,
    [switch]$SkipCausality,
    [switch]$SkipLongRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path $Python)) {
    $Python = "python"
}

function Invoke-LocalMixerCompetition {
    param(
        [string]$Root,
        [int[]]$Seeds,
        [int64]$TargetTokens,
        [int64]$CheckpointInterval
    )

    & .\scripts\run_125m_150m_multiseed_competition.ps1 `
        -Python $Python `
        -DatasetManifest $DatasetManifest `
        -OutputRoot $Root `
        -Seeds $Seeds `
        -TargetTokens $TargetTokens `
        -SeqLen $SeqLen `
        -BatchSize $BatchSize `
        -GradAccumSteps $GradAccumSteps `
        -EvalTokensInterval $EvalTokensInterval `
        -CheckpointTokensInterval $CheckpointInterval `
        -DrmSequenceMode directional_block_cumsum `
        -DrmBlockSize 64 `
        -DrmAndersonIterations 0 `
        -DrmCumsumStepMode velocity `
        -DrmLocalMixer causal_conv `
        -DrmLocalMixerHiddenSize 256 `
        -DrmLocalMixerKernelSize 8 `
        -DrmLocalMixerLayers 2 `
        -DrmLocalMixerScale 0.2
    if ($LASTEXITCODE -ne 0) {
        throw "Local mixer competition failed with exit code $LASTEXITCODE"
    }
}

function Invoke-CausalityChecks {
    param([string]$Root, [int[]]$Seeds)

    foreach ($Seed in $Seeds) {
        $RunName = "drm_125m_real_causal_anderson_b64_stepvelocity_andcandidate_trajectory_s1_mixh256_k8_l2_scale0.2_seed_$Seed"
        $RunDir = Join-Path $Root $RunName
        if (-not (Test-Path $RunDir)) {
            throw "Missing DRM run dir for causality check: $RunDir"
        }
        $Output = Join-Path $RunDir "causality_check.json"
        Write-Host "Checking causal prefix invariance for seed $Seed"
        & $Python scripts\check_125m_local_mixer_causality.py `
            --run-dir $RunDir `
            --output $Output `
            --batch-size $BatchSize `
            --seq-len $SeqLen `
            --device cuda `
            --precision bf16 `
            --drm-block-size 64 `
            --drm-anderson-iterations 0 `
            --drm-cumsum-step-mode velocity `
            --drm-local-mixer causal_conv `
            --drm-local-mixer-hidden-size 256 `
            --drm-local-mixer-kernel-size 8 `
            --drm-local-mixer-layers 2 `
            --drm-local-mixer-scale 0.2
        if ($LASTEXITCODE -ne 0) {
            throw "Causality check failed for seed $Seed with exit code $LASTEXITCODE"
        }
    }
}

Write-Host "125M local mixer validation sequence"
Write-Host "------------------------------------"
Write-Host "confirm_root: $OutputRoot"
Write-Host "confirm_seeds: $($ConfirmSeeds -join ', ')"
Write-Host "confirm_target_tokens: $ConfirmTargetTokens"
Write-Host "long_root: $LongOutputRoot"
Write-Host "long_seeds: $($LongSeeds -join ', ')"
Write-Host "long_target_tokens: $LongTargetTokens"
Write-Host ""

if (-not $SkipConfirm) {
    Invoke-LocalMixerCompetition -Root $OutputRoot -Seeds $ConfirmSeeds -TargetTokens $ConfirmTargetTokens -CheckpointInterval $ConfirmCheckpointTokensInterval
}

if (-not $SkipCausality) {
    Invoke-CausalityChecks -Root $OutputRoot -Seeds $ConfirmSeeds
}

if (-not $SkipLongRun) {
    Invoke-LocalMixerCompetition -Root $LongOutputRoot -Seeds $LongSeeds -TargetTokens $LongTargetTokens -CheckpointInterval $LongCheckpointTokensInterval
}

Write-Host ""
Write-Host "Done."
