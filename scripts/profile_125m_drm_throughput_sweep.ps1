param(
    [string]$OutputRoot = "runs\profile_125m_drm_throughput_sweep",
    [int64]$TargetTokens = 1000000,
    [int]$Seed = 1,
    [int[]]$SeqLens = @(128, 256, 512),
    [ValidateSet("directional_block_cumsum", "directional_superblock_cumsum")]
    [string[]]$SequenceModes = @("directional_block_cumsum"),
    [int[]]$BlockSizes = @(8, 16, 32),
    [int[]]$SuperblockSizes = @(64),
    [int[]]$SuperblockLocalSizes = @(8),
    [int[]]$AndersonIterations = @(1, 2),
    [ValidateSet("candidate", "velocity")]
    [string[]]$CumsumStepModes = @("candidate"),
    [ValidateSet("candidate", "velocity")]
    [string[]]$AndersonTransitionModes = @("candidate"),
    [ValidateSet("trajectory", "endpoint")]
    [string[]]$AndersonScopes = @("trajectory"),
    [int[]]$AndersonBlockStrides = @(1),
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$DatasetManifest = "data\tokens_5b\manifest.json",
    [string]$DrmConfig = "configs\drm_125m_real.yaml",
    [string]$Precision = "bf16",
    [string]$Device = "cuda"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$Rows = @()

foreach ($SeqLen in $SeqLens) {
    foreach ($SequenceMode in $SequenceModes) {
        $LocalSizes = if ($SequenceMode -eq "directional_superblock_cumsum") { $SuperblockLocalSizes } else { $BlockSizes }
        $OuterSizes = if ($SequenceMode -eq "directional_superblock_cumsum") { $SuperblockSizes } else { @(0) }

        foreach ($LocalSize in $LocalSizes) {
            foreach ($OuterSize in $OuterSizes) {
                if ($LocalSize -gt $SeqLen) {
                    continue
                }
                if ($SequenceMode -eq "directional_superblock_cumsum" -and $OuterSize -gt $SeqLen) {
                    continue
                }

                foreach ($Iter in $AndersonIterations) {
                    foreach ($StepMode in $CumsumStepModes) {
                        foreach ($TransitionMode in $AndersonTransitionModes) {
                            foreach ($Scope in $AndersonScopes) {
                                foreach ($BlockStride in $AndersonBlockStrides) {
                                    $BatchSize = if ($SeqLen -le 128) { 8 } elseif ($SeqLen -le 256) { 4 } else { 2 }
                                    $GradAccumSteps = 8
                                    $ModeName = if ($SequenceMode -eq "directional_superblock_cumsum") {
                                        "super${OuterSize}_local${LocalSize}"
                                    } else {
                                        "b${LocalSize}"
                                    }
                                    $RunName = "seq${SeqLen}_${ModeName}_iter${Iter}_step${StepMode}_and${TransitionMode}_${Scope}_s${BlockStride}_seed_${Seed}"
                                    $RunRoot = Join-Path $OutputRoot $RunName

                                    Write-Host ""
                                    Write-Host "Profiling $RunName"

                                    & .\scripts\run_125m_150m_multiseed_competition.ps1 `
                                        -Python $Python `
                                        -DatasetManifest $DatasetManifest `
                                        -DrmConfig $DrmConfig `
                                        -OutputRoot $RunRoot `
                                        -Seeds $Seed `
                                        -TargetTokens $TargetTokens `
                                        -BatchSize $BatchSize `
                                        -GradAccumSteps $GradAccumSteps `
                                        -SeqLen $SeqLen `
                                        -Precision $Precision `
                                        -Device $Device `
                                        -EvalTokensInterval $TargetTokens `
                                        -CheckpointTokensInterval $TargetTokens `
                                        -EvalBatches 1 `
                                        -DrmLogInterval 10 `
                                        -DrmSequenceMode $SequenceMode `
                                        -DrmBlockSize $LocalSize `
                                        -DrmSuperblockSize $OuterSize `
                                        -DrmSuperblockLocalSize $LocalSize `
                                        -DrmAndersonIterations $Iter `
                                        -DrmCumsumStepMode $StepMode `
                                        -DrmAndersonTransitionMode $TransitionMode `
                                        -DrmAndersonScope $Scope `
                                        -DrmAndersonBlockStride $BlockStride `
                                        -SkipGpt2

                                    if ($LASTEXITCODE -ne 0) {
                                        throw "Profile run failed: $RunName"
                                    }

                                    $DrmRunName = if ($SequenceMode -eq "directional_superblock_cumsum") {
                                        "drm_125m_real_causal_anderson_super$($OuterSize)_local$($LocalSize)_step$($StepMode)_and$($TransitionMode)_$($Scope)_s$($BlockStride)_seed_$Seed"
                                    } else {
                                        "drm_125m_real_causal_anderson_b$($LocalSize)_step$($StepMode)_and$($TransitionMode)_$($Scope)_s$($BlockStride)_seed_$Seed"
                                    }
                                    $SummaryPath = Join-Path $RunRoot "$DrmRunName\summary.json"
                                    $MetricsPath = Join-Path $RunRoot "$DrmRunName\metrics_latest.json"
                                    if ((Test-Path $SummaryPath) -and (Test-Path $MetricsPath)) {
                                        $Summary = Get-Content $SummaryPath -Raw | ConvertFrom-Json
                                        $Metrics = Get-Content $MetricsPath -Raw | ConvertFrom-Json
                                        $Latest = $Metrics.latest
                                        $Rows += [pscustomobject]@{
                                            seq_len = $SeqLen
                                            sequence_mode = $SequenceMode
                                            block_size = $LocalSize
                                            superblock_size = $OuterSize
                                            anderson_iterations = $Iter
                                            cumsum_step_mode = $StepMode
                                            anderson_transition_mode = $TransitionMode
                                            anderson_scope = $Scope
                                            anderson_block_stride = $BlockStride
                                            batch_size = $BatchSize
                                            grad_accum_steps = $GradAccumSteps
                                            tokens_seen = [int64]$Summary.tokens_seen
                                            best_val_ce = $Summary.best_val_ce
                                            tokens_per_sec = $Latest.tokens_per_sec
                                            elapsed_sec = $Latest.elapsed_sec
                                            run = $RunName
                                        }
                                        $Rows | Sort-Object -Property tokens_per_sec -Descending |
                                            Export-Csv -Path (Join-Path $OutputRoot "throughput_sweep.csv") -NoTypeInformation
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

Write-Host ""
Write-Host "Sweep complete: $(Join-Path $OutputRoot 'throughput_sweep.csv')"
$Rows | Sort-Object -Property tokens_per_sec -Descending | Format-Table -AutoSize
