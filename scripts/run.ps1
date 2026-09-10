# Unverified on this machine (developed/tested on macOS).
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path .\.venv\Scripts\python.exe)) {
    throw "Missing .venv. Run .\scripts\setup.ps1 first."
}
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
& .\.venv\Scripts\python.exe -m streamlit run (Join-Path (Get-Location) "src\pdm\app.py") `
    --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
