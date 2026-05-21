"""
failure_classifier.py

Classifies LLM output failures into a structured taxonomy with severity
scores calibrated for regulated-industry contexts (finance, compliance, legal).

Failure taxonomy designed from production incident logs, not from what's
easy to detect — prioritising the failures that cause real-world harm.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
from enum import Enum


class FailureSeverity(Enum):
    CRITICAL = "critical"    # Immediate remediation required; may have compliance impact
    HIGH = "high"            # Requires review before next deployment
    MEDIUM = "medium"        # Should be logged and monitored
    LOW = "low"              # Note for next improvement cycle


@dataclass
class FailureType:
    code: str
    name: str
    description: str
    severity: FailureSeverity
    regulated_impact: str    # Why this matters in regulated environments
    detection_hints: List[str]


@dataclass
class ClassifiedFailure:
    failure_code: str
    failure_name: str
    severity: str
    confidence: float          # 0.0–1.0 confidence in this classification
    evidence: List[str]        # Specific text/patterns that triggered this
    regulated_impact: str
    prompt: str
    response: str


# ─────────────────────────────────────────────────────────────
# Failure Taxonomy
# ─────────────────────────────────────────────────────────────

FAILURE_TAXONOMY: Dict[str, FailureType] = {
    "hallucination_factual": FailureType(
        code="hallucination_factual",
        name="Factual Hallucination",
        description="Model states false facts with confidence",
        severity=FailureSeverity.CRITICAL,
        regulated_impact=(
            "In compliance contexts, hallucinated regulatory citations or "
            "policy interpretations can lead to incorrect filings, audit failures, "
            "or actionable advice based on non-existent rules."
        ),
        detection_hints=["cited non-existent", "fabricated", "invented source"],
    ),
    "hallucination_source": FailureType(
        code="hallucination_source",
        name="Source Hallucination",
        description="Model cites sources, documents, or authorities that do not exist",
        severity=FailureSeverity.CRITICAL,
        regulated_impact=(
            "Fabricated regulatory citations or court decisions are a direct "
            "compliance liability if acted upon by regulated entity staff."
        ),
        detection_hints=["sec release", "cfpb guidance", "reg", "section", "act of"],
    ),
    "confidence_miscalibration": FailureType(
        code="confidence_miscalibration",
        name="Confidence Miscalibration",
        description="Model expresses inappropriate certainty or uncertainty",
        severity=FailureSeverity.HIGH,
        regulated_impact=(
            "Overconfident responses on uncertain regulatory interpretations "
            "may lead to decisions without appropriate expert review."
        ),
        detection_hints=["definitely", "certainly", "absolutely", "with certainty"],
    ),
    "refusal_drift": FailureType(
        code="refusal_drift",
        name="Refusal Drift",
        description="Model refuses tasks it previously handled correctly",
        severity=FailureSeverity.HIGH,
        regulated_impact=(
            "Sudden refusal on compliance queries creates workflow gaps — "
            "analysts may not know to escalate if the tool silently stops answering."
        ),
        detection_hints=["i cannot", "i'm unable", "i can't help", "i won't"],
    ),
    "format_degradation": FailureType(
        code="format_degradation",
        name="Format Degradation",
        description="Response structure breaks — expected format (JSON, list, table) not produced",
        severity=FailureSeverity.MEDIUM,
        regulated_impact=(
            "Downstream systems consuming structured model output break silently "
            "when format guarantees degrade — common in automated compliance workflows."
        ),
        detection_hints=["missing structure", "prose instead of list"],
    ),
    "scope_creep": FailureType(
        code="scope_creep",
        name="Scope Creep",
        description="Response answers a different question than asked or significantly expands scope",
        severity=FailureSeverity.MEDIUM,
        regulated_impact=(
            "In document review contexts, scope creep means analysts receive "
            "unrequested analysis that may be mistaken for authoritative guidance."
        ),
        detection_hints=[],
    ),
    "length_collapse": FailureType(
        code="length_collapse",
        name="Length Collapse",
        description="Response is significantly shorter than expected — likely truncated or degraded",
        severity=FailureSeverity.MEDIUM,
        regulated_impact=(
            "Truncated analysis in regulatory contexts may omit material caveats "
            "or required disclosures."
        ),
        detection_hints=[],
    ),
    "instruction_following_failure": FailureType(
        code="instruction_following_failure",
        name="Instruction Following Failure",
        description="Model does not follow explicit instructions in the prompt",
        severity=FailureSeverity.HIGH,
        regulated_impact=(
            "In constrained-output workflows (e.g. produce only JSON, respond only "
            "with specific categories), instruction failures break downstream processing."
        ),
        detection_hints=[],
    ),
}


def classify(prompt: str, response: str) -> List[ClassifiedFailure]:
    """
    Classifies an LLM response against the failure taxonomy.
    Returns a list of detected failures (may be empty if no failures found).
    """
    failures = []
    r_lower = response.lower()

    # Refusal drift detection
    refusal_signals = ["i cannot", "i'm unable", "i can't", "i won't", "i am not able to"]
    refusal_evidence = [s for s in refusal_signals if s in r_lower]
    if refusal_evidence:
        ft = FAILURE_TAXONOMY["refusal_drift"]
        failures.append(ClassifiedFailure(
            failure_code=ft.code,
            failure_name=ft.name,
            severity=ft.severity.value,
            confidence=0.9,
            evidence=refusal_evidence,
            regulated_impact=ft.regulated_impact,
            prompt=prompt,
            response=response,
        ))

    # Overconfidence detection
    overconfidence_signals = ["definitely", "certainly", "absolutely", "with certainty", "i am certain", "guaranteed"]
    oc_evidence = [s for s in overconfidence_signals if s in r_lower]
    if len(oc_evidence) >= 2:
        ft = FAILURE_TAXONOMY["confidence_miscalibration"]
        failures.append(ClassifiedFailure(
            failure_code=ft.code,
            failure_name=ft.name,
            severity=ft.severity.value,
            confidence=0.7,
            evidence=oc_evidence,
            regulated_impact=ft.regulated_impact,
            prompt=prompt,
            response=response,
        ))

    # Length collapse
    word_count = len(response.split())
    prompt_word_count = len(prompt.split())
    if word_count < 5 and prompt_word_count > 20:
        ft = FAILURE_TAXONOMY["length_collapse"]
        failures.append(ClassifiedFailure(
            failure_code=ft.code,
            failure_name=ft.name,
            severity=ft.severity.value,
            confidence=0.95,
            evidence=[f"Response length: {word_count} words"],
            regulated_impact=ft.regulated_impact,
            prompt=prompt,
            response=response,
        ))

    # Format degradation — if prompt requests JSON and response is not
    if "json" in prompt.lower() or "structured" in prompt.lower():
        try:
            import json as json_mod
            json_mod.loads(response)
        except Exception:
            if not response.strip().startswith("{") and not response.strip().startswith("["):
                ft = FAILURE_TAXONOMY["format_degradation"]
                failures.append(ClassifiedFailure(
                    failure_code=ft.code,
                    failure_name=ft.name,
                    severity=ft.severity.value,
                    confidence=0.8,
                    evidence=["Prompt requested JSON/structured output; response is prose"],
                    regulated_impact=ft.regulated_impact,
                    prompt=prompt,
                    response=response,
                ))

    return failures


def severity_score(failures: List[ClassifiedFailure]) -> Tuple[float, str]:
    """
    Returns a composite severity score (0.0–1.0) and top severity label
    for a list of classified failures.
    """
    if not failures:
        return 0.0, "none"

    order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    top = max(failures, key=lambda f: order.get(f.severity, 0))
    score = order.get(top.severity, 0) / 4.0

    return round(score, 2), top.severity
