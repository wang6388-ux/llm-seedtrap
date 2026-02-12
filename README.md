# LLM Seed Trap Project (MVP)

This project studies prompt-induced reasoning instability in large language models under same-instance variation.

---

## Setup

### 1. Create a virtual environment

```powershell
python -m venv .venv
```

### 2. Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Create `.env`

Copy `.env.example` to `.env` and fill in your OpenAI API key:

```text
OPENAI_API_KEY=your_key_here
```

⚠️ Do not commit `.env` to GitHub.

---

## Run Experiments

### 1. Sample model outputs

```powershell
python run_sampling.py
```

### 2. Score and summarize

```powershell
python score_results.py
```

---

## Outputs

After running, the following files will be generated:

- `outputs/raw.jsonl`
- `outputs/scores.csv`
- `outputs/summary.json`

---

## Core Metrics

- Template Attraction Rate
- Internal Inconsistency Rate
- Answer Distribution Entropy
