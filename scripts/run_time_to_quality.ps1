param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$DatasetManifest = "data\tokens\manifest.json",
    [string]$DrmConfig = "configs\drm_125m.yaml",
    [string]$OutputRoot = "runs\time_to_quality_drm_gpt2_37m",
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
    [bool]$RequireGpt2AtLeastDrmTokens = $true
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path $Python)) {
    $Python = "python"
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

function Get-Summary {
    param([string]$RunDir)
    $Path = Join-Path $RunDir "summary.json"
    if (-not (Test-Path $Path)) {
        return $null
    }
    Get-Content $Path -Raw | ConvertFrom-Json
}

function Get-CheckpointResumeValue {
    param([string]$RunDir)
    $LastCheckpoint = Join-Path $RunDir "checkpoint_last.pt"
    if (Test-Path $LastCheckpoint) {
        return $LastCheckpoint
    }
    $Checkpoint = Join-Path $RunDir "checkpoint_latest.pt"
    if (Test-Path $Checkpoint) {
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
        throw "Missing metrics_latest.json for monitor update: $MetricsPath"
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
    param([Nullable[double]]$TargetCe = $null)

    $ArgsList = @(
        "scripts\analyze_time_to_quality.py",
        "--root", $OutputRoot,
        "--target-margin-ce", "$TargetMarginCe",
        "--plateau-window", "$PlateauWindow",
        "--min-improvement-per-million", "$MinImprovementPerMillion"
    )
    if ($null -ne $TargetCe) {
        $ArgsList += @("--target-ce", "$TargetCe")
    }
    & $Python @ArgsList
    if ($LASTEXITCODE -ne 0) {
        throw "Analysis failed with exit code $LASTEXITCODE"
    }
}

function Get-RunStatus {
    param([string]$RunName)

    $StatusPath = Join-Path $OutputRoot "time_to_quality_status.json"
    if (-not (Test-Path $StatusPath)) {
        return $null
    }
    $Status = Get-Content $StatusPath -Raw | ConvertFrom-Json
    foreach ($Run in $Status.runs) {
        if ($Run.run -eq $RunName) {
            return $Run
        }
    }
    return $null
}

function Get-MaxTokensForFamily {
    param([string]$Family)

    $StatusPath = Join-Path $OutputRoot "time_to_quality_status.json"
    if (-not (Test-Path $StatusPath)) {
        return 0L
    }
    $Status = Get-Content $StatusPath -Raw | ConvertFrom-Json
    $MaxSeen = 0L
    foreach ($Run in $Status.runs) {
        if ($Run.family -eq $Family -and $null -ne $Run.tokens_seen) {
            $MaxSeen = [math]::Max($MaxSeen, [int64]$Run.tokens_seen)
        }
    }
    return $MaxSeen
}

function Invoke-DrmUntil {
    param(
        [int]$Seed,
        [int64]$TargetTokens
    )

    $RunName = "drm_37m_causal_anderson_b8_seed_$Seed"
    $RunDir = Join-Path $OutputRoot $RunName
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
    if ($Resume -ne "") {
        $ArgsList += @("--resume", $Resume)
    }

    Write-Host "Running DRM seed $Seed until $TargetTokens tokens"
    & $Python @ArgsList
    if ($LASTEXITCODE -ne 0) {
        throw "DRM run failed with exit code $LASTEXITCODE for seed $Seed"
    }
    Update-Monitor -RunDir $RunDir -Family "drm" -Seed $Seed
}

function Invoke-Gpt2Until {
    param(
        [int]$Seed,
        [int64]$TargetTokens
    )

    $RunName = "gpt2_36m_seed_$Seed"
    $RunDir = Join-Path $OutputRoot $RunName
    $Resume = Get-CheckpointResumeValue -RunDir $RunDir

    $ArgsList = @(
        "scripts\train_gpt2_memmap.py",
        "--model-size", "gpt2_125m",
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

    Write-Host "Running GPT-2 seed $Seed until $TargetTokens tokens"
    & $Python @ArgsList
    if ($LASTEXITCODE -ne 0) {
        throw "GPT-2 run failed with exit code $LASTEXITCODE for seed $Seed"
    }
    Update-Monitor -RunDir $RunDir -Family "gpt2" -Seed $Seed
}

foreach ($Seed in $Seeds) {
    $RunName = "drm_37m_causal_anderson_b8_seed_$Seed"
    $RunDir = Join-Path $OutputRoot $RunName
    $Target = $InitialTargetTokens
    while ($true) {
        $Summary = Get-Summary -RunDir $RunDir
        if (($null -eq $Summary) -or ([int64]$Summary.tokens_seen -lt $Target)) {
            Invoke-DrmUntil -Seed $Seed -TargetTokens $Target
        } else {
            Write-Host "Reusing DRM seed $Seed at $($Summary.tokens_seen) tokens"
            Update-Monitor -RunDir $RunDir -Family "drm" -Seed $Seed
        }

        Invoke-Analysis
        $RunStatus = Get-RunStatus -RunName $RunName
        if (($null -ne $RunStatus) -and ([bool]$RunStatus.plateau_detected)) {
            Write-Host "DRM seed $Seed plateau detected."
            break
        }

        $Summary = Get-Summary -RunDir $RunDir
        if (($null -ne $Summary) -and ([int64]$Summary.tokens_seen -ge $MaxTokens)) {
            Write-Host "DRM seed $Seed reached MaxTokens."
            break
        }
        $Target = [math]::Min($MaxTokens, [int64]$Summary.tokens_seen + $ChunkTokens)
    }
}

Invoke-Analysis
$StatusPath = Join-Path $OutputRoot "time_to_quality_status.json"
$Status = Get-Content $StatusPath -Raw | ConvertFrom-Json
$TargetCe = [double]$Status.target_ce
$DrmTokenFloor = Get-MaxTokensForFamily -Family "drm"
Write-Host "Global target CE from DRM plateau/current curves: $TargetCe"
if ($RequireGpt2AtLeastDrmTokens) {
    Write-Host "GPT-2 plateau stop is disabled before $DrmTokenFloor tokens, unless target CE is reached first."
}

foreach ($Seed in $Seeds) {
    $RunName = "gpt2_36m_seed_$Seed"
    $RunDir = Join-Path $OutputRoot $RunName
    $Target = $InitialTargetTokens
    while ($true) {
        $Summary = Get-Summary -RunDir $RunDir
        if (($null -eq $Summary) -or ([int64]$Summary.tokens_seen -lt $Target)) {
            Invoke-Gpt2Until -Seed $Seed -TargetTokens $Target
        } else {
            Write-Host "Reusing GPT-2 seed $Seed at $($Summary.tokens_seen) tokens"
            Update-Monitor -RunDir $RunDir -Family "gpt2" -Seed $Seed
        }

        Invoke-Analysis -TargetCe $TargetCe
        $RunStatus = Get-RunStatus -RunName $RunName
        if (($null -ne $RunStatus) -and ([bool]$RunStatus.target_reached)) {
            Write-Host "GPT-2 seed $Seed reached target CE."
            break
        }
        $Summary = Get-Summary -RunDir $RunDir
        $Gpt2TokensSeen = if ($null -ne $Summary) { [int64]$Summary.tokens_seen } else { 0L }
        $CanStopForPlateau = (-not $RequireGpt2AtLeastDrmTokens) -or ($Gpt2TokensSeen -ge $DrmTokenFloor)
        if (($null -ne $RunStatus) -and ([bool]$RunStatus.plateau_detected) -and $CanStopForPlateau) {
            Write-Host "GPT-2 seed $Seed plateau detected before target CE."
            break
        }
        if (($null -ne $RunStatus) -and ([bool]$RunStatus.plateau_detected) -and (-not $CanStopForPlateau)) {
            Write-Host "GPT-2 seed $Seed plateau signal ignored until it reaches DRM token floor $DrmTokenFloor."
        }

        if (($null -ne $Summary) -and ([int64]$Summary.tokens_seen -ge $MaxTokens)) {
            Write-Host "GPT-2 seed $Seed reached MaxTokens before target CE."
            break
        }
        $Target = [math]::Min($MaxTokens, [int64]$Summary.tokens_seen + $ChunkTokens)
    }
}

Invoke-Analysis -TargetCe $TargetCe
Write-Host "Dashboard: $(Join-Path $OutputRoot 'dashboard.html')"
Write-Host "Status:    $(Join-Path $OutputRoot 'time_to_quality_status.json')"
