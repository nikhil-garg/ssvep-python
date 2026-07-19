$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$baseCheckpoints = Join-Path $root 'outputs\experiments\resonate_and_fire_deep_gain_search\checkpoints'
$endpointOut = Join-Path $root 'outputs\experiments\resonate_and_fire_decision_endpoints'
New-Item -ItemType Directory -Force -Path $endpointOut | Out-Null
while ((Get-ChildItem $baseCheckpoints -Filter '*.npz' -ErrorAction SilentlyContinue).Count -lt 150) {
    Start-Sleep -Seconds 60
}
$python = Join-Path $root '.venv\Scripts\python.exe'
$script = Join-Path $root 'scripts\run_resonate_and_fire_endpoint_extension.py'
& $python $script 1> (Join-Path $endpointOut 'endpoint_search.stdout.log') 2> (Join-Path $endpointOut 'endpoint_search.stderr.log')
