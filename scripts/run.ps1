# Written for Windows PowerShell (Win10/11). Full Windows execution is not done on the macOS development machine.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path .\.venv\Scripts\python.exe)) {
    throw "Missing .venv. Run .\scripts\setup.ps1 first."
}
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
& .\.venv\Scripts\python.exe -m streamlit run (Join-Path (Get-Location) "src\pdm\app.py") `
    --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
