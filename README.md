# LLM Seed Trap Project (MVP)

## Setup
1. Create a virtual env
   - Windows: python -m venv .venv
   - Activate:
     - PowerShell: .\.venv\Scripts\Activate.ps1

2. Install deps
   pip install -r requirements.txt

3. Create .env
   Copy .env.example -> .env and fill OPENAI_API_KEY.

## Run
1) Sample model outputs
   python run_sampling.py

2) Score + summarize
   python score_results.py

Outputs:
- outputs/raw.jsonl
- outputs/scores.csv
- outputs/summary.json
