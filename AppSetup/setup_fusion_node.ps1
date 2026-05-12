<#
.SYNOPSIS
    Bootstrap a fresh Windows node for the Fusion Centers FL Update.

.DESCRIPTION
    Provisions, in order:
      1. A local .venv\ at the project root.
      2. requirements_fusion.txt into that venv.
      3. The UCI Communities and Crime (Unnormalized) data file under
         $HOME\datasets\CommunitiesCrime\.
      4. The .names schema file via `python -m ...generate_names_file`.
      5. (Optional, -Verify) Runs `pytest tests/unit -q -k fusion`
         to confirm the install actually works.

    Re-runs are safe: existing venv / data files are reused, not clobbered.

    Prerequisite: Python 3.9+ on PATH (use `python --version` to check).

.PARAMETER Verify
    After install, run `pytest tests/unit -q -k fusion` against the venv.

.PARAMETER SkipData
    Don't touch $HOME\datasets\CommunitiesCrime -- assumes dataset is
    already in place (e.g. mounted volume).

.PARAMETER VenvDir
    Where to put the virtualenv. Defaults to <project root>\.venv.

.PARAMETER DataDir
    Where to put the UCI dataset. Defaults to $HOME\datasets\CommunitiesCrime.

.EXAMPLE
    pwsh -File AppSetup\setup_fusion_node.ps1

.EXAMPLE
    pwsh -File AppSetup\setup_fusion_node.ps1 -Verify

.EXAMPLE
    pwsh -File AppSetup\setup_fusion_node.ps1 -SkipData -VenvDir D:\envs\fusion
#>
[CmdletBinding()]
param(
    [switch]$Verify,
    [switch]$SkipData,
    [string]$VenvDir = "",
    [string]$DataDir = "",
    [string]$PythonBin = "python"
)

$ErrorActionPreference = "Stop"

# ── Resolve paths ───────────────────────────────────────────────────────
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$RequirementsFile = Join-Path $ScriptDir "requirements_fusion.txt"

if (-not $VenvDir) { $VenvDir = Join-Path $ProjectRoot ".venv" }
if (-not $DataDir) { $DataDir = Join-Path $HOME "datasets\CommunitiesCrime" }

$UciZipUrl = "https://archive.ics.uci.edu/static/public/211/communities+and+crime+unnormalized.zip"

function Log  ([string]$msg) { Write-Host "`n[setup] $msg" -ForegroundColor Cyan }
function Warn ([string]$msg) { Write-Host "`n[warn]  $msg" -ForegroundColor Yellow }
function Fail ([string]$msg) { Write-Host "`n[fail]  $msg" -ForegroundColor Red; exit 1 }

# ── 1. Check Python ─────────────────────────────────────────────────────
# Probe a candidate `python`-style command and return the reported version
# if it actually runs (or $null if it's the Windows app-execution-alias stub
# / a broken install). Used to validate $PythonBin and to try `py` as a
# fallback below.
function Test-PythonCmd {
    param([string]$Bin)
    $cmd = Get-Command $Bin -ErrorAction SilentlyContinue
    if (-not $cmd) { return $null }
    $out = (& $Bin -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrEmpty($out) -or $out -notmatch '^\d+\.\d+$') {
        return $null
    }
    return [pscustomobject]@{ Bin = $Bin; Version = $out; Source = $cmd.Source }
}

$pyInfo = Test-PythonCmd $PythonBin

# If the user didn't override -PythonBin and the default `python` resolved
# to the Microsoft Store stub, transparently fall back to the `py` launcher.
# `py.exe` ships at C:\Windows\py.exe and is immune to the app-execution
# -alias PATH-shadowing issue.
if (-not $pyInfo -and $PythonBin -eq "python") {
    Warn "python on PATH is the Microsoft Store stub (or otherwise broken). Trying 'py' launcher..."
    $pyInfo = Test-PythonCmd "py"
    if ($pyInfo) { $PythonBin = "py" }
}

if (-not $pyInfo) {
    Fail @"
python (or py) on PATH could not be invoked to report a version.

This is almost always the Windows Microsoft Store stub (it prints
"Python was not found, run without arguments to install from the
Microsoft Store..." and exits without producing output), or a broken
install with no usable python.exe / py.exe on PATH.

Fixes (any one is enough):
  1. Install real Python 3.9+ via:
       winget install --id Python.Python.3.12 --source winget
     then close and reopen this PowerShell terminal so PATH is picked up.
  2. Disable the Microsoft Store alias:
       Settings -> Apps -> Advanced app settings -> App execution aliases
       -> toggle off python.exe and python3.exe.
  3. Re-run this script with the py launcher explicitly:
       powershell -ExecutionPolicy Bypass -File AppSetup\setup_fusion_node.ps1 -Verify -PythonBin py
"@
}
$pyVer = $pyInfo.Version
Log "Using Python $pyVer from $($pyInfo.Source)  (-PythonBin $PythonBin)"

# ── 2. Create venv ──────────────────────────────────────────────────────
if (-not (Test-Path $VenvDir)) {
    Log "Creating venv at $VenvDir..."
    & $PythonBin -m venv $VenvDir
} else {
    Log "Reusing existing venv at $VenvDir."
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$VenvPip    = Join-Path $VenvDir "Scripts\pip.exe"
if (-not (Test-Path $VenvPython)) {
    Fail "venv looks broken (no Scripts\python.exe at $VenvDir). Delete and re-run."
}

# ── 3. Python deps ──────────────────────────────────────────────────────
Log "Upgrading pip + installing $RequirementsFile..."
& $VenvPython -m pip install --upgrade pip wheel setuptools
& $VenvPip install -r $RequirementsFile
if ($LASTEXITCODE -ne 0) { Fail "pip install failed (exit $LASTEXITCODE)." }

# ── 4. Dataset acquisition ──────────────────────────────────────────────
if ($SkipData) {
    Log "Skipping dataset acquisition (-SkipData)."
} else {
    if (-not (Test-Path $DataDir)) { New-Item -ItemType Directory -Path $DataDir -Force | Out-Null }
    $DataFile  = Join-Path $DataDir "CommViolPredUnnormalizedData.txt"
    $NamesFile = Join-Path $DataDir "communities_and_crime_unnormalized.names"

    if (Test-Path $DataFile) {
        Log "Data file already present at $DataFile -- skipping download."
    } else {
        Log "Downloading UCI Communities and Crime (Unnormalized)..."
        $TmpZip     = New-TemporaryFile
        $TmpZipFile = "$($TmpZip.FullName).zip"
        Move-Item $TmpZip.FullName $TmpZipFile -Force
        $TmpExtract = Join-Path $env:TEMP ("comm_crime_" + [guid]::NewGuid().ToString("N"))
        New-Item -ItemType Directory -Path $TmpExtract -Force | Out-Null

        try {
            Invoke-WebRequest -Uri $UciZipUrl -OutFile $TmpZipFile -UseBasicParsing
            Expand-Archive -Path $TmpZipFile -DestinationPath $TmpExtract -Force

            $found = Get-ChildItem -Path $TmpExtract -Filter "CommViolPredUnnormalizedData.txt" -Recurse |
                Select-Object -First 1
            if (-not $found) { Fail "Could not find CommViolPredUnnormalizedData.txt in the UCI zip." }
            Move-Item -Path $found.FullName -Destination $DataFile -Force
            $dataSize = (Get-Item $DataFile).Length
            Log "Saved $DataFile ($dataSize bytes)"
        } finally {
            if (Test-Path $TmpZipFile)  { Remove-Item $TmpZipFile  -Force -ErrorAction SilentlyContinue }
            if (Test-Path $TmpExtract)  { Remove-Item $TmpExtract  -Recurse -Force -ErrorAction SilentlyContinue }
        }
    }

    if (Test-Path $NamesFile) {
        Log ".names schema already present -- skipping generation."
    } else {
        Log "Generating .names schema via ucimlrepo..."
        Push-Location $ProjectRoot
        try {
            & $VenvPython -m Config.DatasetConfig.CommunitiesCrime_Sampling.generate_names_file `
                --output $NamesFile
            if ($LASTEXITCODE -ne 0) { Fail "generate_names_file failed (exit $LASTEXITCODE)." }
        } finally {
            Pop-Location
        }
    }

    # ── 5. Quick schema validation ──────────────────────────────────────
    Log "Validating dataset schema..."
    $validatorScript = @"
from pathlib import Path
from Config.DatasetConfig.CommunitiesCrime_Sampling.commCrimeDatasetLoad import (
    parse_names_file, load_raw, CRIME_RATE_COLUMNS, SENSITIVE_COLUMNS,
)
d = Path(r'$DataDir')
names = parse_names_file(d / 'communities_and_crime_unnormalized.names')
df = load_raw(d / 'CommViolPredUnnormalizedData.txt', names)
print(f'  rows={len(df)}  cols={df.shape[1]}')
mc = [c for c in CRIME_RATE_COLUMNS if c not in df.columns]
ms = [c for c in SENSITIVE_COLUMNS  if c not in df.columns]
if mc:
    print(f'  !! missing crime-rate cols: {mc}')
else:
    print(f'  ok - all {len(CRIME_RATE_COLUMNS)} crime-rate columns present')
if ms:
    print(f'  note: sensitive columns absent (will be silently skipped): {ms}')
"@
    $tmpValidator = New-TemporaryFile
    $tmpValidatorPy = "$($tmpValidator.FullName).py"
    Move-Item $tmpValidator.FullName $tmpValidatorPy -Force
    Set-Content -Path $tmpValidatorPy -Value $validatorScript -Encoding UTF8
    try {
        Push-Location $ProjectRoot
        & $VenvPython $tmpValidatorPy
        if ($LASTEXITCODE -ne 0) { Fail "Schema validation failed (exit $LASTEXITCODE)." }
    } finally {
        Pop-Location
        Remove-Item $tmpValidatorPy -Force -ErrorAction SilentlyContinue
    }
}

# ── 6. Optional: run test suite ─────────────────────────────────────────
if ($Verify) {
    Log "Running test suite (tests/unit -k fusion) to verify install..."
    Push-Location $ProjectRoot
    try {
        & $VenvPython -m pytest "tests/unit" -q -k fusion
        if ($LASTEXITCODE -ne 0) { Fail "pytest reported failures (exit $LASTEXITCODE)." }
    } finally {
        Pop-Location
    }
}

Log "Done. Activate the venv with:"
Write-Host "    & `"$VenvDir\Scripts\Activate.ps1`""
Log "Then follow DeveloperDocs\RUNNING_FUSION_EXPERIMENTS.md for the experiment matrix."
