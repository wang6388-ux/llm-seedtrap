import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import sympy as sp

RAW_PATH = Path("outputs/raw.jsonl")
DATA_PATH = Path("data/seed_trap_set.json")
SCORES_CSV = Path("outputs/scores.csv")
SUMMARY_JSON = Path("outputs/summary.json")

# --------- Answer extraction ---------

FRAC_RE = re.compile(r"(-?\d+)\s*/\s*(-?\d+)")
INT_RE = re.compile(r"(?<![\d/])-?\d+(?![\d/])")

def normalize_answer(ans: str) -> str | None:
    if ans is None:
        return None
    ans = ans.strip()
    # reduce fraction or integer via sympy Rational
    try:
        if "/" in ans:
            a, b = ans.split("/")
            r = sp.Rational(int(a.strip()), int(b.strip()))
            return f"{int(r.p)}" if r.q == 1 else f"{int(r.p)}/{int(r.q)}"
        else:
            r = sp.Rational(int(ans), 1)
            return f"{int(r.p)}"
    except Exception:
        return None

def extract_final_answer(text: str) -> str | None:
    # Heuristic: take the LAST fraction if any; else the LAST integer.
    if not text:
        return None
    fracs = FRAC_RE.findall(text)
    if fracs:
        a, b = fracs[-1]
        return normalize_answer(f"{a}/{b}")
    ints = INT_RE.findall(text)
    if ints:
        return normalize_answer(ints[-1])
    return None

# --------- Internal inconsistency detection ---------

# Find candidate arithmetic expressions (very conservative)
EXPR_RE = re.compile(r"([0-9\(\)\s\+\-\*\/\^\!]+)")

def to_sympy_expr(expr_str: str) -> sp.Expr | None:
    s = expr_str.strip()
    if len(s) < 5:
        return None
    # normalize
    s = s.replace("^", "**")
    # convert factorial like "7!" -> "factorial(7)"
    s = re.sub(r"(\d+)\s*!", r"factorial(\1)", s)
    # reject if contains letters (we want purely numeric)
    if re.search(r"[A-Za-z]", s):
        return None
    try:
        return sp.sympify(s, locals={"factorial": sp.factorial})
    except Exception:
        return None

def detect_inconsistency(output_text: str, final_answer: str | None) -> tuple[int, str | None]:
    """
    Returns (inconsistent_flag, evidence_expr)
    inconsistent if we find a numeric expression in the text that evaluates
    to a rational number different from final_answer.
    """
    if final_answer is None or not output_text:
        return (0, None)

    try:
        final_val = sp.Rational(final_answer)
    except Exception:
        return (0, None)

    # collect candidate expressions
    candidates = []
    for m in EXPR_RE.finditer(output_text):
        chunk = m.group(1)
        # only consider chunks with at least one operator or factorial
        if ("*" not in chunk and "/" not in chunk and "!" not in chunk and "^" not in chunk and "**" not in chunk):
            continue
        expr = to_sympy_expr(chunk)
        if expr is None:
            continue
        candidates.append((chunk.strip(), expr))

    # evaluate and compare
    for raw, expr in candidates:
        try:
            val = sp.nsimplify(expr)
            # if integer/ratio
            if isinstance(val, (sp.Integer, sp.Rational)):
                if sp.Rational(val) != final_val:
                    return (1, raw)
        except Exception:
            continue

    return (0, None)

# --------- Entropy ---------

def entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    h = 0.0
    for v in counter.values():
        p = v / total
        h -= p * math.log(p + 1e-12)
    return float(h)

# --------- Main scoring ---------

def load_seed_index():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    idx = {}
    for item in data:
        idx[item["id"]] = {
            "correct_answer": normalize_answer(item["correct_answer"]),
            "template_answers": [normalize_answer(x) for x in item.get("template_answers", [])],
        }
    return idx

def read_raw():
    rows = []
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def main():
    seed_idx = load_seed_index()
    rows = read_raw()

    scored = []
    for r in rows:
        pid = r["problem_id"]
        meta = seed_idx.get(pid, {})
        correct = meta.get("correct_answer")
        templates = set([t for t in meta.get("template_answers", []) if t is not None])

        out = r.get("output_text", "")
        final_ans = extract_final_answer(out)

        acc = int(final_ans is not None and correct is not None and final_ans == correct)
        template_hit = int(final_ans is not None and final_ans in templates)

        inconsistent, evidence = detect_inconsistency(out, final_ans)

        scored.append({
            **r,
            "final_answer": final_ans,
            "is_correct": acc,
            "is_template_hit": template_hit,
            "is_inconsistent": inconsistent,
            "inconsistency_evidence": evidence,
        })

    df = pd.DataFrame(scored)
    SCORES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SCORES_CSV, index=False)

    # Summary per (problem_id, prompt_type)
    summary = []
    for (pid, ptype), g in df.groupby(["problem_id", "prompt_type"]):
        answers = [a for a in g["final_answer"].tolist() if a is not None]
        h = entropy(Counter(answers))
        summary.append({
            "problem_id": pid,
            "prompt_type": ptype,
            "n": int(len(g)),
            "accuracy": float(g["is_correct"].mean()),
            "template_attraction_rate": float(g["is_template_hit"].mean()),
            "internal_inconsistency_rate": float(g["is_inconsistent"].mean()),
            "answer_entropy": h,
            "top_answers": Counter(answers).most_common(5),
        })

    out = {
        "summary_rows": summary,
        "notes": {
            "entropy": "Natural log entropy over extracted final answers per group.",
            "inconsistency": "Flags if any numeric expression in text evaluates to a value different from final answer (heuristic).",
        }
    }

    SUMMARY_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {SCORES_CSV} and {SUMMARY_JSON}")

if __name__ == "__main__":
    main()
