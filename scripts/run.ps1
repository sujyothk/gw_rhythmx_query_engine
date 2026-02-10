param(
  [string]$VenvDir = ".venv",
  [string]$IndexPath = "artifacts\index.pkl",
  [string]$DataDir = "data",
  [string]$ModelPath = "models\tinyllama",
  [string]$Query = "",
  [switch]$Ui,
  [int]$TopK = 8,
  [double]$Temperature = 0.2,
  [int]$MaxTokens = 700,
  [switch]$ShowContext
)

$ErrorActionPreference = "Stop"

try {
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force | Out-Null
} catch {}

# Activate venv
$activate = Join-Path $VenvDir "Scripts\Activate.ps1"
if (!(Test-Path $activate)) {
  throw "Virtual environment not found. Run: .\scripts\setup.ps1"
}
. $activate

# Build index if missing (and data exists)
if (!(Test-Path $IndexPath)) {
  if (!(Test-Path $DataDir)) {
    throw "Index not found and data dir missing: $DataDir. Copy FHIR JSON files to $DataDir and run setup."
  }
  $jsonCount = (Get-ChildItem -Path $DataDir -Filter *.json -ErrorAction SilentlyContinue | Measure-Object).Count
  if ($jsonCount -eq 0) {
    throw "Index not found and no *.json files in $DataDir. Copy FHIR JSON files there and rerun."
  }
  New-Item -ItemType Directory -Force -Path (Split-Path $IndexPath) | Out-Null
  python cli.py build-index --data-dir $DataDir --index-path $IndexPath
}

if ($Ui) {
  streamlit run app.py
  exit 0
}

if ([string]::IsNullOrWhiteSpace($Query)) {
  throw "Provide -Query (for CLI mode) or use -Ui for the Streamlit app."
}

$args = @(
  "cli.py", "ask",
  "--index-path", $IndexPath,
  "--query", $Query,
  "--top-k", $TopK.ToString(),
  "--use-llm",
  "--llm-provider", "hf",
  "--llm-model-path", $ModelPath,
  "--temperature", $Temperature.ToString(),
  "--max-tokens", $MaxTokens.ToString()
)

if ($ShowContext) {
  $args += "--show-context"
}

python @args
