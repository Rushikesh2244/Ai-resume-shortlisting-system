"""
utils/report.py
Generates a ranked shortlist report:
  - Pretty console table (colorama + tabulate)
  - CSV output to /output/
  - Detailed per-candidate breakdown
"""

import os
import csv
from datetime import datetime
from typing import List, Dict, Any

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    COLOR = True
except ImportError:
    COLOR = False

try:
    from tabulate import tabulate
    TABULATE = True
except ImportError:
    TABULATE = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _color(text: str, color_code: str) -> str:
    if COLOR:
        return color_code + text + Style.RESET_ALL
    return text


def _bar(score: float, width: int = 20) -> str:
    """ASCII progress bar for a score in [0, 100]."""
    filled = int((score / 100) * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {score:.1f}%"


def _medal(rank: int) -> str:
    return {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")


# ── Console Report ────────────────────────────────────────────────────────────

def print_summary_table(results: List[Dict[str, Any]], jd_title: str = "") -> None:
    """Print a compact ranked summary table to stdout."""
    sep = "─" * 80
    print()
    print(_color(sep, Fore.CYAN if COLOR else ""))
    title = f"  RESUME SHORTLISTING RESULTS  |  JD: {jd_title}" if jd_title else "  RESUME SHORTLISTING RESULTS"
    print(_color(title, Fore.CYAN if COLOR else ""))
    print(_color(sep, Fore.CYAN if COLOR else ""))
    print(f"  Candidates evaluated : {len(results)}")
    print(f"  Report generated     : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(_color(sep, Fore.CYAN if COLOR else ""))
    print()

    headers = ["Rank", "Name", "Score", "Skills Match", "Experience", "Degree", "Email"]
    rows = []
    for r in results:
        edu = r.get("education", {})
        skill_pct = f"{r['scores']['skill_match']*100:.0f}%  ({len(r['matched_skills'])}/{len(r['matched_skills'])+len(r['missing_skills'])} skills)"
        rows.append([
            _medal(r["rank"]),
            r["name"],
            _bar(r["final_score"]),
            skill_pct,
            f"{r['experience_years']} yrs",
            edu.get("degree", "N/A"),
            r["email"] or "—",
        ])

    if TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    else:
        print("  |  ".join(headers))
        for row in rows:
            print("  |  ".join(str(c) for c in row))
    print()


def print_detailed_breakdown(results: List[Dict[str, Any]]) -> None:
    """Print per-candidate detailed breakdown."""
    sep = "─" * 60
    for r in results:
        print(_color(sep, Fore.YELLOW if COLOR else ""))
        print(_color(f"  {_medal(r['rank'])}  {r['name']}", Fore.WHITE if COLOR else ""))
        print(_color(sep, Fore.YELLOW if COLOR else ""))

        s = r["scores"]
        print(f"  Overall Score      : {_color(_bar(r['final_score']), Fore.GREEN if COLOR else '')}")
        print(f"  TF-IDF Cosine      : {s['tfidf_cosine']*100:.1f}%  (weight 40%)")
        print(f"  Skill Match        : {s['skill_match']*100:.1f}%  (weight 35%)")
        print(f"  Education          : {s['education']*100:.1f}%  (weight 10%)")
        print(f"  Experience         : {s['experience']*100:.1f}%  (weight 15%)")
        print()

        edu = r.get("education", {})
        print(f"  Degree             : {edu.get('degree', 'N/A')}")
        if edu.get("gpa"):
            print(f"  GPA                : {edu['gpa']}/10")
        print(f"  Experience Years   : {r['experience_years']}")
        print(f"  Email              : {r['email'] or '—'}")
        print()

        if r["matched_skills"]:
            matched_str = ", ".join(r["matched_skills"][:10])
            if len(r["matched_skills"]) > 10:
                matched_str += f" ... (+{len(r['matched_skills'])-10} more)"
            print(f"  ✅ Matched Skills  : {_color(matched_str, Fore.GREEN if COLOR else '')}")
        if r["missing_skills"]:
            missing_str = ", ".join(r["missing_skills"][:8])
            if len(r["missing_skills"]) > 8:
                missing_str += f" ... (+{len(r['missing_skills'])-8} more)"
            print(f"  ❌ Missing Skills  : {_color(missing_str, Fore.RED if COLOR else '')}")
        print()


# ── CSV Export ────────────────────────────────────────────────────────────────

def export_csv(results: List[Dict[str, Any]], output_dir: str, jd_title: str = "") -> str:
    """Save results to a CSV file. Returns the output path."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = jd_title.replace(" ", "_").replace("/", "-") if jd_title else "results"
    filename = f"shortlist_{safe_title}_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)

    fieldnames = [
        "rank", "name", "email", "phone",
        "final_score", "tfidf_cosine", "skill_match", "education_score", "experience_score",
        "degree", "gpa", "experience_years",
        "matched_skills", "missing_skills", "total_skills_found",
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            edu = r.get("education", {})
            s = r["scores"]
            writer.writerow({
                "rank":              r["rank"],
                "name":              r["name"],
                "email":             r["email"],
                "phone":             r["phone"],
                "final_score":       r["final_score"],
                "tfidf_cosine":      round(s["tfidf_cosine"] * 100, 2),
                "skill_match":       round(s["skill_match"] * 100, 2),
                "education_score":   round(s["education"] * 100, 2),
                "experience_score":  round(s["experience"] * 100, 2),
                "degree":            edu.get("degree", ""),
                "gpa":               edu.get("gpa", ""),
                "experience_years":  r["experience_years"],
                "matched_skills":    "; ".join(r["matched_skills"]),
                "missing_skills":    "; ".join(r["missing_skills"]),
                "total_skills_found": len(r.get("skills", [])),
            })

    print(_color(f"\n  📄 CSV report saved → {filepath}\n", Fore.CYAN if COLOR else ""))
    return filepath
