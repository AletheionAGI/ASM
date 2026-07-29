param(
    [string]$OutputRoot = "runs\probe_125m_superblock_suite",
    [int[]]$Seeds = @(1),
    [int64[]]$TargetTokensList = @(1000000, 10000000),
    [int]$SeqLen = 512,
    [int]$BatchSize = 2,
    [int]$GradAccumSteps = 8,
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$DatasetManifest = "data\tokens_5b\manifest.json",
    [string]$DrmConfig = "configs\drm_125m_real.yaml",
    [string]$Precision = "bf16",
    [string]$Device = "cuda",
    [int]$EvalBatches = 4,
    [switch]$DryRun,
    [switch]$DryRunForward
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$DryRunArgs = @{}
if ($DryRun) {
    $DryRunArgs["DryRun"] = $true
}
if ($DryRunForward) {
    $DryRunArgs["DryRunForward"] = $true
}

foreach ($TargetTokens in $TargetTokensList) {
    $TokenLabel = if ($TargetTokens -ge 1000000) {
        "$([int]($TargetTokens / 1000000))m"
    } else {
        "$($TargetTokens)"
    }

    Write-Host ""
    Write-Host "125M probe suite target=$TargetTokens tokens"
    Write-Host "----------------------------------------"

    & .\scripts\run_125m_150m_multiseed_competition.ps1 `
        -Python $Python `
        -DatasetManifest $DatasetManifest `
        -DrmConfig $DrmConfig `
        -OutputRoot (Join-Path $OutputRoot "${TokenLabel}_drm_b8_iter2") `
        -Seeds $Seeds `
        -TargetTokens $TargetTokens `
        -SeqLen $SeqLen `
        -BatchSize $BatchSize `
        -GradAccumSteps $GradAccumSteps `
        -Precision $Precision `
        -Device $Device `
        -EvalTokensInterval $TargetTokens `
        -CheckpointTokensInterval $TargetTokens `
        -EvalBatches $EvalBatches `
        -DrmSequenceMode directional_block_cumsum `
        -DrmBlockSize 8 `
        -DrmAndersonIterations 2 `
        -DrmCumsumStepMode candidate `
        -DrmAndersonTransitionMode candidate `
        -SkipGpt2 `
        @DryRunArgs
    if ($LASTEXITCODE -ne 0) {
        throw "DRM b8 iter2 probe failed"
    }

    & .\scripts\run_125m_150m_multiseed_competition.ps1 `
        -Python $Python `
        -DatasetManifest $DatasetManifest `
        -DrmConfig $DrmConfig `
        -OutputRoot (Join-Path $OutputRoot "${TokenLabel}_drm_b32_iter1") `
        -Seeds $Seeds `
        -TargetTokens $TargetTokens `
        -SeqLen $SeqLen `
        -BatchSize $BatchSize `
        -GradAccumSteps $GradAccumSteps `
        -Precision $Precision `
        -Device $Device `
        -EvalTokensInterval $TargetTokens `
        -CheckpointTokensInterval $TargetTokens `
        -EvalBatches $EvalBatches `
        -DrmSequenceMode directional_block_cumsum `
        -DrmBlockSize 32 `
        -DrmAndersonIterations 1 `
        -DrmCumsumStepMode candidate `
        -DrmAndersonTransitionMode candidate `
        -SkipGpt2 `
        @DryRunArgs
    if ($LASTEXITCODE -ne 0) {
        throw "DRM b32 iter1 probe failed"
    }

    & .\scripts\run_125m_150m_multiseed_competition.ps1 `
        -Python $Python `
        -DatasetManifest $DatasetManifest `
        -DrmConfig $DrmConfig `
        -OutputRoot (Join-Path $OutputRoot "${TokenLabel}_drm_super64_local8_iter2") `
        -Seeds $Seeds `
        -TargetTokens $TargetTokens `
        -SeqLen $SeqLen `
        -BatchSize $BatchSize `
        -GradAccumSteps $GradAccumSteps `
        -Precision $Precision `
        -Device $Device `
        -EvalTokensInterval $TargetTokens `
        -CheckpointTokensInterval $TargetTokens `
        -EvalBatches $EvalBatches `
        -DrmSequenceMode directional_superblock_cumsum `
        -DrmSuperblockSize 64 `
        -DrmSuperblockLocalSize 8 `
        -DrmBlockSize 8 `
        -DrmAndersonIterations 2 `
        -DrmCumsumStepMode velocity `
        -DrmAndersonTransitionMode velocity `
        -SkipGpt2 `
        @DryRunArgs
    if ($LASTEXITCODE -ne 0) {
        throw "DRM superblock probe failed"
    }

    & .\scripts\run_125m_150m_multiseed_competition.ps1 `
        -Python $Python `
        -DatasetManifest $DatasetManifest `
        -DrmConfig $DrmConfig `
        -OutputRoot (Join-Path $OutputRoot "${TokenLabel}_gpt2_125m") `
        -Seeds $Seeds `
        -TargetTokens $TargetTokens `
        -SeqLen $SeqLen `
        -BatchSize $BatchSize `
        -GradAccumSteps $GradAccumSteps `
        -Precision $Precision `
        -Device $Device `
        -EvalTokensInterval $TargetTokens `
        -CheckpointTokensInterval $TargetTokens `
        -EvalBatches $EvalBatches `
        -SkipDrm `
        @DryRunArgs
    if ($LASTEXITCODE -ne 0) {
        throw "GPT-2 probe failed"
    }
}

Write-Host ""
Write-Host "Probe suite complete: $OutputRoot"
