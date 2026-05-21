"""
drift_monitor.py

Detects statistically significant shifts in LLM output behaviour across
model versions, prompt changes, or deployment updates.

Compares output distributions from a baseline snapshot against a current
run across a standardised prompt battery.
"""

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict


@dataclass
class DriftResult:
    prompt_id: str
    baseline_response: str
    current_response: str
    similarity_score: float      # 0.0–1.0, higher = more similar
    drift_detected: bool
    drift_severity: str          # "none" | "minor" | "moderate" | "critical"
    failure_types: List[str]
    timestamp: str


@dataclass
class DriftReport:
    run_id: str
    baseline_version: str
    current_version: str
    total_prompts: int
    drifted_prompts: int
    drift_rate: float
    critical_count: int
    moderate_count: int
    minor_count: int
    results: List[DriftResult]
    timestamp: str
    passed: bool                 # False if drift_rate exceeds threshold


# Severity thresholds — calibrated for regulated-industry tolerances
DRIFT_THRESHOLDS = {
    "similarity_minor":    0.75,   # below this = minor drift
    "similarity_moderate": 0.50,   # below this = moderate drift
    "similarity_critical": 0.25,   # below this = critical drift
    "acceptable_drift_rate": 0.10, # >10% of prompts drifting = report failure
    "critical_tolerance":  0.02,   # >2% critical drifts = immediate flag
}


def compute_token_similarity(a: str, b: str) -> float:
    """
    Token-level Jaccard similarity between two responses.
    Fast, interpretable baseline — sufficient for detecting
    structural/content drift without requiring embeddings.
    """
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def classify_failure_types(baseline: str, current: str) -> List[str]:
    """
    Identifies which failure modes are present in the drift.
    Returns a list of failure type labels.
    """
    failures = []
    b_lower = baseline.lower()
    c_lower = current.lower()

    # Refusal drift: model now refuses where it previously answered
    refusal_signals = ["i cannot", "i'm unable", "i can't", "i won't", "i am not able"]
    baseline_refused = any(s in b_lower for s in refusal_signals)
    current_refused = any(s in c_lower for s in refusal_signals)
    if current_refused and not baseline_refused:
        failures.append("refusal_drift")
    if baseline_refused and not current_refused:
        failures.append("refusal_reversal")

    # Format degradation: structural changes (markdown, lists, headers)
    baseline_has_structure = any(c in baseline for c in ["**", "##", "- ", "1. "])
    current_has_structure = any(c in current for c in ["**", "##", "- ", "1. "])
    if baseline_has_structure and not current_has_structure:
        failures.append("format_degradation")

    # Length drift: response significantly shorter or longer
    len_ratio = len(current.split()) / max(len(baseline.split()), 1)
    if len_ratio < 0.4:
        failures.append("length_collapse")
    elif len_ratio > 2.5:
        failures.append("verbosity_expansion")

    # Confidence miscalibration: hedging language appearing/disappearing
    hedge_words = ["might", "may", "could", "possibly", "perhaps", "uncertain", "unclear"]
    baseline_hedges = sum(1 for w in hedge_words if w in b_lower)
    current_hedges = sum(1 for w in hedge_words if w in c_lower)
    if current_hedges > baseline_hedges + 3:
        failures.append("confidence_miscalibration_high_hedge")
    elif baseline_hedges > current_hedges + 3:
        failures.append("confidence_miscalibration_low_hedge")

    return failures


def classify_severity(similarity: float, failure_types: List[str]) -> str:
    """
    Maps similarity score and failure types to a severity level.
    Critical failures in regulated contexts take precedence over score alone.
    """
    high_severity_failures = {"refusal_drift", "length_collapse", "confidence_miscalibration_high_hedge"}

    if any(f in high_severity_failures for f in failure_types):
        if similarity < DRIFT_THRESHOLDS["similarity_moderate"]:
            return "critical"
        return "moderate"

    if similarity >= DRIFT_THRESHOLDS["similarity_minor"]:
        return "none"
    if similarity >= DRIFT_THRESHOLDS["similarity_moderate"]:
        return "minor"
    if similarity >= DRIFT_THRESHOLDS["similarity_critical"]:
        return "moderate"
    return "critical"


def compare_responses(
    prompt_id: str,
    baseline_response: str,
    current_response: str,
) -> DriftResult:
    similarity = compute_token_similarity(baseline_response, current_response)
    failure_types = classify_failure_types(baseline_response, current_response)
    severity = classify_severity(similarity, failure_types)

    return DriftResult(
        prompt_id=prompt_id,
        baseline_response=baseline_response,
        current_response=current_response,
        similarity_score=round(similarity, 4),
        drift_detected=severity != "none",
        drift_severity=severity,
        failure_types=failure_types,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def run_drift_check(
    baseline: Dict[str, str],
    current: Dict[str, str],
    baseline_version: str = "baseline",
    current_version: str = "current",
) -> DriftReport:
    """
    Compares two sets of prompt→response mappings and produces a drift report.

    Args:
        baseline: Dict mapping prompt_id → response (from previous snapshot)
        current:  Dict mapping prompt_id → response (from current run)
        baseline_version: Label for the baseline (e.g. model version or date)
        current_version:  Label for the current run
    """
    run_id = hashlib.sha256(
        f"{baseline_version}{current_version}{datetime.now().isoformat()}".encode()
    ).hexdigest()[:12]

    results = []
    for prompt_id, baseline_resp in baseline.items():
        current_resp = current.get(prompt_id, "")
        result = compare_responses(prompt_id, baseline_resp, current_resp)
        results.append(result)

    total = len(results)
    drifted = [r for r in results if r.drift_detected]
    critical = [r for r in drifted if r.drift_severity == "critical"]
    moderate = [r for r in drifted if r.drift_severity == "moderate"]
    minor = [r for r in drifted if r.drift_severity == "minor"]

    drift_rate = len(drifted) / max(total, 1)
    critical_rate = len(critical) / max(total, 1)

    passed = (
        drift_rate <= DRIFT_THRESHOLDS["acceptable_drift_rate"]
        and critical_rate <= DRIFT_THRESHOLDS["critical_tolerance"]
    )

    return DriftReport(
        run_id=run_id,
        baseline_version=baseline_version,
        current_version=current_version,
        total_prompts=total,
        drifted_prompts=len(drifted),
        drift_rate=round(drift_rate, 4),
        critical_count=len(critical),
        moderate_count=len(moderate),
        minor_count=len(minor),
        results=results,
        timestamp=datetime.now(timezone.utc).isoformat(),
        passed=passed,
    )


def save_report(report: DriftReport, output_path: str = "outputs/drift_report.json") -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(asdict(report), f, indent=2)
    print(f"Drift report saved: {output_path}")


def print_summary(report: DriftReport) -> None:
    status = "PASS ✅" if report.passed else "FAIL ❌"
    print(f"\n{'='*60}")
    print(f"  Drift Check: {status}")
    print(f"  Run ID:      {report.run_id}")
    print(f"  Baseline:    {report.baseline_version}")
    print(f"  Current:     {report.current_version}")
    print(f"{'='*60}")
    print(f"  Prompts:     {report.total_prompts}")
    print(f"  Drifted:     {report.drifted_prompts} ({report.drift_rate:.1%})")
    print(f"  Critical:    {report.critical_count}")
    print(f"  Moderate:    {report.moderate_count}")
    print(f"  Minor:       {report.minor_count}")
    print(f"{'='*60}\n")

    if report.critical_count > 0:
        print("  ⚠️  Critical drifts detected:")
        for r in report.results:
            if r.drift_severity == "critical":
                print(f"    [{r.prompt_id}] failures: {r.failure_types}")
        print()
