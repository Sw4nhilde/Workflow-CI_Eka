$ErrorActionPreference = 'Stop'

$env:MLFLOW_TRACKING_URI = 'https://dagshub.com/Sw4nhilde/Membangun_model_Muhammad-Eka-Mandiri-Sujanto.mlflow'
$runId = (Get-Content -Path (Join-Path $PSScriptRoot 'latest_run_id.txt') -Raw).Trim()

$env:RUN_ID = $runId
python (Join-Path $PSScriptRoot 'build_docker_helper.py')