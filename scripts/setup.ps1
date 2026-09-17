# Written for Windows PowerShell (Win10/11). Full Windows execution is not done on the macOS development machine.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

function Get-PythonMinorVersion {
    param(
        [Parameter(Mandatory = $true)][string]$Exe,
        [string[]]$PrefixArgs = @()
    )
    $invokeArgs = @()
    if ($PrefixArgs) { $invokeArgs += $PrefixArgs }
    $invokeArgs += @("-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    $output = & $Exe @invokeArgs
    if ($LASTEXITCODE -ne 0) { return $null }
    $line = @($output) | Where-Object { $_ -and "$_".Trim() } | Select-Object -Last 1
    if (-not $line) { return $null }
    return ([string]$line).Trim()
}

$candidates = @(
    @{ Exe = "py"; Args = @("-3.12") },
    @{ Exe = "py"; Args = @("-3.11") },
    @{ Exe = "python3.12"; Args = @() },
    @{ Exe = "python3.11"; Args = @() },
    @{ Exe = "python"; Args = @() }
)

$chosenExe = $null
$chosenArgs = @()
$chosenVer = $null
$probed = @()

foreach ($cand in $candidates) {
    try {
        $found = Get-Command $cand.Exe -ErrorAction SilentlyContinue
        if (-not $found) { continue }
        $source = [string]$found.Source
        if (
            $cand.Exe -eq "python" -and
            @($cand.Args).Count -eq 0 -and
            $source -match "WindowsApps"
        ) {
            $probed += "python skipped (WindowsApps stub)"
            continue
        }
        $ver = Get-PythonMinorVersion -Exe $cand.Exe -PrefixArgs $cand.Args
        $label = ($cand.Exe + " " + ($cand.Args -join " ")).Trim()
        if ($ver) { $probed += "$label -> $ver" }
        if ($ver -eq "3.12" -or $ver -eq "3.11") {
            $chosenExe = $cand.Exe
            $chosenArgs = @($cand.Args)
            $chosenVer = $ver
            break
        }
    } catch {
        continue
    }
}

if (-not $chosenExe) {
    $detail = if ($probed) { " Probed: " + ($probed -join "; ") + "." } else { "" }
    throw "Python 3.11 or 3.12 not found.$detail Install 3.11 or 3.12, then re-run."
}

Write-Host "Using $chosenExe $($chosenArgs -join ' ') (Python $chosenVer)"

$venvArgs = @()
if ($chosenArgs) { $venvArgs += $chosenArgs }
$venvArgs += @("-m", "venv", ".venv")
& $chosenExe @venvArgs
if ($LASTEXITCODE -ne 0) { throw "Failed to create .venv with $chosenExe" }

$venvPy = Join-Path (Get-Location) ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    throw "venv created but .venv\Scripts\python.exe is missing"
}

function Invoke-VenvPython {
    param([Parameter(Mandatory = $true)][string[]]$PyArgs)
    & $venvPy @PyArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed ($LASTEXITCODE): .venv\Scripts\python.exe $($PyArgs -join ' ')"
    }
}

Invoke-VenvPython -PyArgs @("-m", "pip", "install", "--upgrade", "pip", "wheel")
Invoke-VenvPython -PyArgs @("-m", "pip", "install", "-r", "requirements-lock.txt")
Invoke-VenvPython -PyArgs @("-m", "pip", "install", "-e", ".")
Invoke-VenvPython -PyArgs @("-m", "pdm", "doctor")

Write-Host "Setup complete. Launch with: .\scripts\run.ps1"
