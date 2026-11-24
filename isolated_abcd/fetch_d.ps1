$ErrorActionPreference = 'Stop'
$revision = '532794c51e046a0ddaf560e7972789688d46bf63'
$dir = Join-Path $PSScriptRoot 'local/cross'
New-Item -ItemType Directory -Force -Path $dir | Out-Null
foreach ($split in 'train','test') {
    $name = "$split-00000-of-00001.parquet"
    $file = Join-Path $dir $name
    if (Test-Path -LiteralPath $file) { continue }
    $tmp = "$file.partial"
    Invoke-WebRequest -Uri "https://huggingface.co/datasets/HikaruAsano/cross-psych-101/resolve/$revision/data/$name" -OutFile $tmp
    Move-Item -LiteralPath $tmp -Destination $file
}
Set-Content -LiteralPath (Join-Path $dir 'REVISION') -Value $revision
