param([ValidateSet('full_context','rewrite','rag')][string]$Memory='full_context', [int]$Tasks=56)
$ErrorActionPreference='Stop'
$env:PYTHONUTF8='1'
if (!$env:SII_REASONING_EFFORT) { $env:SII_REASONING_EFFORT='none' }
$taskRoot=Split-Path $PSScriptRoot -Parent
if (!$env:SII_API_KEY) {
 if ($env:SII_GATEWAY_TOKEN) { $env:SII_API_KEY=$env:SII_GATEWAY_TOKEN }
 else {
  $tokenFile=if ($env:CDCA_SII_TOKEN_FILE) {$env:CDCA_SII_TOKEN_FILE} else {'C:/npx/token.json'}
  if (Test-Path -LiteralPath $tokenFile) { $env:SII_API_KEY=(Get-Content -LiteralPath $tokenFile -Raw | ConvertFrom-Json).token }
 }
}
if (!$env:SII_API_KEY -or !$env:SII_MODEL) { throw 'SII token or SII_MODEL is missing.' }
$env:VITA_MODEL_CONFIG_PATH=Join-Path $taskRoot 'configs/vita.yaml'
$env:VITA_DATA_DIR=Join-Path $taskRoot 'upstream/VitaBench-2.0/data'
$target=Join-Path $env:VITA_DATA_DIR 'vita/domains/personalization'
New-Item -ItemType Directory -Force $target | Out-Null
Copy-Item (Join-Path $taskRoot 'data/vita/tasks.json') $target
Push-Location (Join-Path $taskRoot 'upstream/VitaBench-2.0')
try {
 & "$taskRoot/.venv/Scripts/vita.exe" run --domain personalization --memory-type $Memory --agent-llm $env:SII_MODEL --user-llm $env:SII_MODEL --evaluator-llm $env:SII_MODEL --num-tasks $Tasks --num-trials 4 --max-steps 50 --max-concurrency 2 --enable-outcome-reward --save-to "reproduction/$($env:SII_REASONING_EFFORT)/$Memory.json"
 if ($LASTEXITCODE) { throw 'Vita evaluation failed' }
} finally { Pop-Location }
