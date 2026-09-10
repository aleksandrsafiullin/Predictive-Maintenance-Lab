# Unverified on this machine (developed/tested on macOS). Run from the project root.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$py = $null
foreach ($cand in @("py -3.11", "py -3.12", "python")) {
    try {
        $ver = Invoke-Expression "$cand -c `"import sys; print(sys.version)`""
        if ($LASTEXITCODE -eq 0) { $py = $cand; break }
    } catch { }
}
if (-not $py) { throw "Python 3.11+ not found" }

Write-Host "Using $py"
Invoke-Expression "$py -m venv .venv"
.\.venv\Scripts\python.exe -m pip install --upgrade pip wheel
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pdm doctor
Write-Host "Setup complete. Launch with: .\scripts\run.ps1"
