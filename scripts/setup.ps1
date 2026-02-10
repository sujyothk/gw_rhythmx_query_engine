param(
  [string]$PythonExe = "python",
  [string]$VenvDir = ".venv",
  [string]$ModelId = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  [string]$ModelOutDir = "models\tinyllama",
  [string]$DataDir = "data",
  [string]$IndexPath = "artifacts\index.pkl"
)

$ErrorActionPreference = "Stop"

Write-Host "=== Rhythmx Setup (HF Transformers Local LLM) ==="
Write-Host "Project root: $(Get-Location)"

# Allow script execution in this PowerShell session only
try {
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force | Out-Null
} catch {
  # If the environment disallows it, user can still run manually.
}

# 1) Create venv if missing
if (!(Test-Path $VenvDir)) {
  Write-Host "Creating virtual environment at $VenvDir ..."
  & $PythonExe -m venv $VenvDir
} else {
  Write-Host "Virtual environment already exists: $VenvDir"
}

# 2) Activate venv
$activate = Join-Path $VenvDir "Scripts\Activate.ps1"
if (!(Test-Path $activate)) {
  throw "Could not find venv activate script at: $activate. Did venv creation fail?"
}
. $activate

# 3) Upgrade pip and install requirements
Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# 4) Ensure folders exist
New-Item -ItemType Directory -Force -Path (Split-Path $IndexPath) | Out-Null
New-Item -ItemType Directory -Force -Path $ModelOutDir | Out-Null

# 5) Download model (one-time)
if (!(Test-Path (Join-Path $ModelOutDir "config.json"))) {
  Write-Host "Downloading model: $ModelId to $ModelOutDir ..."
  python .\scripts\download_hf_model.py --model-id $ModelId --out-dir $ModelOutDir
} else {
  Write-Host "Model folder already looks populated: $ModelOutDir"
}

# 6) Build index (only if data dir has json)
if (!(Test-Path $DataDir)) {
  Write-Host "WARNING: Data directory not found: $DataDir"
  Write-Host "Create it and copy your FHIR JSON files (*.json) there."
} else {
  $jsonCount = (Get-ChildItem -Path $DataDir -Filter *.json -ErrorAction SilentlyContinue | Measure-Object).Count
  if ($jsonCount -eq 0) {
    Write-Host "WARNING: No *.json files found in $DataDir. Copy FHIR JSON files there, then re-run setup or build index manually."
  } else {
    Write-Host "Building retrieval index from $DataDir ..."
    python cli.py build-index --data-dir $DataDir --index-path $IndexPath
  }
}

Write-Host ""
Write-Host "Setup complete."
Write-Host "Next:"
Write-Host "  - Ask a question:"
Write-Host ("      .\scripts\run.ps1 -Query ""Does the patient have asthma?"" -ModelPath """ + $ModelOutDir + """ -IndexPath """ + $IndexPath + """")
Write-Host "  - Or start UI:"
Write-Host ("      .\scripts\run.ps1 -Ui -ModelPath """ + $ModelOutDir + """ -IndexPath """ + $IndexPath + """")
