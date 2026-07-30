param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$RunRoot = "runs\competition_125m_local_mixer_h256_l2_s02_150m",
    [ValidateSet(1, 2, 3)]
    [int]$Seed = 1,
    [string]$Checkpoint = "",
    [string]$Device = "cuda",
    [string]$Dtype = "auto",
    [int]$MaxNewTokens = 160,
    [double]$Temperature = 0.8,
    [int]$TopK = 40,
    [int]$MaxTurns = 3,
    [int]$MaxPromptTokens = 512,
    [switch]$ShowPrompt
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path $Python)) {
    $Python = "python"
}

if (-not (Test-Path $RunRoot)) {
    throw "Run root not found: $RunRoot"
}

$RunDir = Get-ChildItem -Path $RunRoot -Directory |
    Where-Object { $_.Name -eq "gpt2_125m_real_seed_$Seed" } |
    Select-Object -First 1

if ($null -eq $RunDir) {
    throw "GPT-2 seed $Seed run directory not found under $RunRoot"
}

if (-not $Checkpoint) {
    foreach ($Name in @("checkpoint_best.pt", "checkpoint_last.pt", "checkpoint_latest.pt")) {
        $Candidate = Join-Path $RunDir.FullName $Name
        if (Test-Path $Candidate) {
            $Checkpoint = $Candidate
            break
        }
    }
}

if (-not $Checkpoint) {
    throw "No GPT-2 checkpoint found for seed $Seed in $($RunDir.FullName)"
}

$ChatArgs = @(
    "scripts\chat_gpt2_125m_4090_base.py",
    "--run-dir", $RunDir.FullName,
    "--checkpoint", $Checkpoint,
    "--device", $Device,
    "--dtype", $Dtype,
    "--max-new-tokens", "$MaxNewTokens",
    "--max-prompt-tokens", "$MaxPromptTokens",
    "--temperature", "$Temperature",
    "--top-k", "$TopK",
    "--max-turns", "$MaxTurns"
)

if ($ShowPrompt) {
    $ChatArgs += "--show-prompt"
}

& $Python @ChatArgs
