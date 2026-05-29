"""
models/scorer.py
Scores and ranks resumes against a job description using:
  1. TF-IDF + Cosine Similarity  (semantic text overlap)
  2. Skill Match Ratio           (required skills coverage)
  3. Education Level Bonus       (degree weight)
  4. Experience Bonus            (years of experience)

Final score is a weighted combination of all four signals.
"""

from typing import List, Dict, Any, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.parser import clean_text
from utils.ner_extractor import SKILLS_VOCAB, extract_skills


# ── Configurable weights (must sum to 1.0) ────────────────────────────────────
WEIGHTS = {
    "tfidf_cosine":   0.40,   # broad semantic overlap with JD
    "skill_match":    0.35,   # % of JD skills found in resume
    "education":      0.10,   # degree level bonus
    "experience":     0.15,   # years of relevant experience
}

# Education scoring map (degree_level → normalized 0-1 score)
EDUCATION_SCORE_MAP = {
    0: 0.0,   # no degree detected
    1: 0.2,   # high school / unknown
    2: 0.4,   # diploma
    3: 0.7,   # bachelor's
    4: 0.9,   # master's
    5: 1.0,   # PhD
}

MAX_EXPERIENCE_YEARS = 10.0   # cap for normalization


def _tfidf_cosine(jd_text: str, resume_texts: List[str]) -> List[float]:
    """
    Fit a TF-IDF vectorizer on the JD + all resumes, then compute
    cosine similarity of each resume against the JD.
    Returns a list of similarity scores in [0, 1].
    """
    cleaned = [clean_text(t) for t in [jd_text] + resume_texts]
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),     # unigrams + bigrams catch "machine learning", "natural language", etc.
        stop_words="english",
        max_features=5000,
        sublinear_tf=True,      # log-scale TF dampens common terms
    )
    tfidf_matrix = vectorizer.fit_transform(cleaned)
    jd_vec = tfidf_matrix[0]            # first row = JD
    resume_vecs = tfidf_matrix[1:]      # remaining rows = resumes
    scores = cosine_similarity(jd_vec, resume_vecs).flatten()
    return scores.tolist()


def _skill_match_score(jd_text: str, resume_skills: List[str]) -> float:
    """
    Extract skills mentioned in JD, compute what fraction are present in resume.
    Returns a float in [0, 1].
    """
    jd_skills = extract_skills(jd_text)
    if not jd_skills:
        return 0.5   # no skill info in JD → neutral score
    matched = set(resume_skills) & set(jd_skills)
    return len(matched) / len(jd_skills)


def _education_score(degree_level: int) -> float:
    return EDUCATION_SCORE_MAP.get(degree_level, 0.0)


def _experience_score(years: float) -> float:
    return min(years / MAX_EXPERIENCE_YEARS, 1.0)


def score_resumes(
    jd_text: str,
    candidates: List[Dict[str, Any]],   # list of parsed resume dicts
    resume_texts: List[str],            # corresponding raw texts
) -> List[Dict[str, Any]]:
    """
    Score and rank all candidates against a job description.

    Args:
        jd_text:       raw text of the job description
        candidates:    list of dicts from ner_extractor.extract_all()
        resume_texts:  list of raw resume texts (same order as candidates)

    Returns:
        List of result dicts sorted by final_score descending, including
        breakdown of individual score components.
    """
    assert len(candidates) == len(resume_texts), "Mismatch between candidates and texts."

    # ── 1. TF-IDF cosine similarity ──────────────────────────────────────────
    cosine_scores = _tfidf_cosine(jd_text, resume_texts)

    results = []
    for i, (candidate, raw_text) in enumerate(zip(candidates, resume_texts)):
        edu     = candidate.get("education", {})
        skills  = candidate.get("skills", [])
        exp_yrs = candidate.get("experience_years", 0.0)

        # ── 2. Individual component scores ───────────────────────────────────
        s_tfidf = cosine_scores[i]
        s_skill = _skill_match_score(jd_text, skills)
        s_edu   = _education_score(edu.get("degree_level", 0))
        s_exp   = _experience_score(exp_yrs)

        # ── 3. Weighted final score ───────────────────────────────────────────
        final = (
            WEIGHTS["tfidf_cosine"] * s_tfidf +
            WEIGHTS["skill_match"]  * s_skill +
            WEIGHTS["education"]    * s_edu   +
            WEIGHTS["experience"]   * s_exp
        )

        # ── 4. Matched / missing skills breakdown ────────────────────────────
        jd_skills = set(extract_skills(jd_text))
        matched_skills = sorted(set(skills) & jd_skills)
        missing_skills = sorted(jd_skills - set(skills))

        results.append({
            **candidate,
            "scores": {
                "tfidf_cosine":  round(s_tfidf, 4),
                "skill_match":   round(s_skill, 4),
                "education":     round(s_edu,   4),
                "experience":    round(s_exp,   4),
                "final":         round(final,   4),
            },
            "final_score":      round(final * 100, 2),   # percentage
            "matched_skills":   matched_skills,
            "missing_skills":   missing_skills,
            "rank":             None,   # assigned after sorting
        })

    # ── 5. Sort by final score descending ─────────────────────────────────────
    results.sort(key=lambda x: x["final_score"], reverse=True)
    for rank, r in enumerate(results, start=1):
        r["rank"] = rank

    return results
