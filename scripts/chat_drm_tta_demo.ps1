param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$RunDir = "runs\compare_10m_seed1\drm_37m_causal_anderson_b8_seed_1",
    [string]$Checkpoint = "",
    [string]$Device = "cuda",
    [string]$Dtype = "auto",
    [int]$MaxNewTokens = 120,
    [double]$Temperature = 0.75,
    [int]$TopK = 40,
    [int]$MaxTurns = 3,
    [int]$MaxPromptTokens = 512,
    [int]$Seed = 1,
    [switch]$ShowPrompt
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path $Python)) {
    $Python = "python"
}

$ChatArgs = @(
    "scripts\chat_drm_125m_4090_base.py",
    "--run-dir", $RunDir,
    "--device", $Device,
    "--dtype", $Dtype,
    "--max-new-tokens", "$MaxNewTokens",
    "--max-prompt-tokens", "$MaxPromptTokens",
    "--temperature", "$Temperature",
    "--top-k", "$TopK",
    "--max-turns", "$MaxTurns",
    "--seed", "$Seed"
)

if ($Checkpoint) {
    $ChatArgs += @("--checkpoint", $Checkpoint)
}
if ($ShowPrompt) {
    $ChatArgs += @("--show-prompt")
}

& $Python @ChatArgs
