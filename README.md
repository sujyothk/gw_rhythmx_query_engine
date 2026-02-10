# Rhythmx Query Engine (FHIR) — Local LLM, No Paid APIs

This submission implements a **FHIR query engine** that prioritizes **accuracy and validation transparency** while using a **local LLM** (no paid APIs).

## Required Output Format (enforced)

Each answer includes:
- **Clinical facts** retrieved from the knowledge base
- **FHIR resource IDs** as evidence
- **Relevant medical code references** (ICD-10, SNOMED, LOINC, RxNorm, etc.) when available

Example:
```
The patient has Type 2 Diabetes Mellitus (ICD-10: E11.9)
The most recent HbA1c (LOINC: 4548-4) value is 8.2%

Source: Condition/123, Observation/907
```

If the information is not present, the engine responds exactly:
```
Not found in provided records.
```

---

## Approach and design choices

### 1) FHIR normalization (structured extraction)
FHIR JSON files are parsed and normalized into structured tables:
- Conditions
- MedicationStatements
- Observations (labs)
- AllergyIntolerance
- Encounters

This enables high-precision extraction of clinical facts and medical codes.

### 2) Retrieval layer (hybrid lexical retrieval)
To support broad/natural questions and provide evidence, the engine indexes “chunks” derived from normalized resources using:
- **word TF-IDF** (general text matching)
- **character n-gram TF-IDF** (robust matching for codes like ICD/LOINC/RxNorm and typos)

### 3) Answering strategy (accuracy-first + local LLM)
- Deterministic handlers extract facts and codes where possible.
- A **local LLM** formats the final response using *only* the extracted facts + retrieved context.
- Guardrails:
  - If the local LLM output does not include a final `Source:` line with FHIR IDs, the engine falls back to deterministic formatting.

This keeps answers grounded, readable, and easy to validate.

---

## Libraries, tools, and models used

### Core libraries
- `python-dateutil` for date parsing
- `scikit-learn` for TF-IDF and similarity
- `numpy` for vector operations
- `streamlit` for the optional UI

### Local LLM runtime (pip-only)
- `torch` + `transformers` for local generation
- `huggingface-hub` for one-time model download
- `safetensors` for safe weight loading

### Default model
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (CPU-friendly)

You can swap to a larger local model (3B–7B) if you have more RAM, but CPU inference will be slower.

---

## Assumptions and limitations
- Input directory contains FHIR JSON files (`*.json`). Bundles are supported.
- Focused extraction for common resource types:
  - `Condition`, `MedicationStatement`, `Observation`, `AllergyIntolerance`, `Encounter`
- Some datasets may store clinical concepts in other resources (e.g., `MedicationRequest`, `DiagnosticReport`). Those are not deeply parsed but are still indexed as fallback chunks.
- Local LLM output quality depends on the chosen model. Guardrails prevent unsupported sources.

---

## One-command setup/run (Windows + macOS)

### Windows (PowerShell)
1) Copy FHIR JSON files into `data\`
2) Run setup:
```powershell
.\scripts\setup.ps1
```
3) Run a query:
```powershell
.\scripts\run.ps1 -Query "Does the patient have asthma?" -ShowContext
```
4) Start UI:
```powershell
.\scripts\run.ps1 -Ui
```

(If you prefer CMD, use `scripts\setup.cmd` and `scripts\run.cmd`.)

### macOS / Linux (bash)
1) Copy FHIR JSON files into `data/`
2) Run setup:
```bash
chmod +x scripts/setup.sh scripts/run.sh
./scripts/setup.sh
```
3) Run a query:
```bash
./scripts/run.sh --query "Does the patient have asthma?" --show-context
```
4) Start UI:
```bash
./scripts/run.sh --ui
```

---

## Batch evaluation output (validator-friendly)
This generates `artifacts/eval_results.json`:
```bash
python cli.py eval --data-dir ./data --questions ./questions.json --use-llm --llm-provider hf --llm-model-path ./models/tinyllama --out ./artifacts/eval_results.json
```
