param([ValidateSet('null','groundtruth','full_context','rewrite','rag')][string]$Memory='full_context', [int]$Tasks=56, [int]$Trials=4, [string[]]$TaskIds=@(), [string]$RunName='medium-task16384-prep-none-v4-retry1', [switch]$LocalEmbeddings, [int]$Concurrency=2)
$ErrorActionPreference='Stop'
$env:PYTHONUTF8='1'
$env:SII_REASONING_EFFORT='medium'
$env:SII_MODEL='nex-n2.5-pro'
$env:VITA_STRICT_MEMORY='1'
if ($LocalEmbeddings) {
 $env:NO_PROXY='127.0.0.1,localhost'
 $env:no_proxy='127.0.0.1,localhost'
 $env:VITA_EMBEDDING_URL='http://127.0.0.1:8766/v1'
 $env:VITA_EMBEDDING_KEY='local-only'
 $env:VITA_EMBEDDING_MODEL='paraphrase-multilingual-MiniLM-L12-v2'
 $env:VITA_EMBEDDING_MAX_CONCURRENCY='4'
}
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
$sourceData=Join-Path $taskRoot 'data/vita/tasks.json'
$runtimeData=Join-Path $target 'tasks.json'
if (!(Test-Path $runtimeData) -or (Get-FileHash $sourceData).Hash -ne (Get-FileHash $runtimeData).Hash) { Copy-Item $sourceData $runtimeData }
Push-Location (Join-Path $taskRoot 'upstream/VitaBench-2.0')
try {
 $selection=@('--num-tasks', $Tasks)
 if ($TaskIds.Count -gt 0) { $selection=@('--task-ids')+$TaskIds }
 $variant=$Memory
 if ($Memory -eq 'rag' -and $LocalEmbeddings) { $variant='rag-local-minilm' }
 $runPath=Join-Path $env:VITA_DATA_DIR "simulations/verification/$RunName"
 if ($env:VITA_DIAGNOSTIC_2048 -ne '1') {
  $policyPath=Join-Path $runPath 'run-policy.json'
  if ((Test-Path $runPath) -and !(Test-Path $policyPath) -and (Get-ChildItem -LiteralPath $runPath -Filter '*.json')) { throw 'Existing directory contains results from a different inference policy; choose a new RunName.' }
  New-Item -ItemType Directory -Force $runPath | Out-Null
  if (Test-Path $policyPath) {
   $savedPolicy=Get-Content -LiteralPath $policyPath -Raw | ConvertFrom-Json
   if ($savedPolicy.protocol_revision -ne 'split-budget-prep-none-v4' -or $savedPolicy.track_retry_limit -ne 1) { throw 'Existing results use a different protocol; choose a new RunName.' }
  } else { @{protocol_revision='split-budget-prep-none-v4';track_retry_limit=1;model='nex-n2.5-pro';reasoning_effort='medium';preparation_effort='none';task_token_budget=16384;budget_scope='subtask_conversation';preparation_token_budget=$null;fallback=$null;scoring_token_budget=$null} | ConvertTo-Json | Set-Content -Encoding utf8 -LiteralPath $policyPath }
 }
 & "$taskRoot/.venv/Scripts/vita.exe" run --domain personalization --memory-type $Memory --agent-llm $env:SII_MODEL --user-llm $env:SII_MODEL --evaluator-llm $env:SII_MODEL @selection --num-trials $Trials --max-steps 50 --max-concurrency $Concurrency --save-to "verification/$RunName/$variant.json"
 if ($LASTEXITCODE) { throw 'Vita evaluation failed' }
} finally { Pop-Location }
