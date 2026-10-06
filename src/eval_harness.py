"""
eval_harness.py

Regression test harness for LLM systems.

Runs a curated set of "golden queries" (prompt + expected output criteria)
against the live model and checks whether responses still meet the defined
quality criteria. Designed to run on every deployment as a gate check.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Callable, Optional
from dataclasses import dataclass, asdict, field


@dataclass
class GoldenQuery:
    """
    A golden query is a prompt with defined acceptance criteria.
    The criteria are functions that take the model response and return bool.
    """
    id: str
    prompt: str
    category: str               # e.g. "factual", "refusal", "format", "reasoning"
    severity: str               # "critical" | "high" | "medium" | "low"
    description: str            # what this query is testing
    must_contain: List[str] = field(default_factory=list)    # substrings that must appear
    must_not_contain: List[str] = field(default_factory=list)  # substrings that must not appear
    min_length: int = 0
    max_length: int = 10000
    expected_format: Optional[str] = None    # "json" | "list" | "prose"


@dataclass
class QueryResult:
    query_id: str
    prompt: str
    response: str
    passed: bool
    failures: List[str]
    latency_ms: float
    severity: str
    timestamp: str


@dataclass
class HarnessReport:
    run_id: str
    model_version: str
    total: int
    passed: int
    failed: int
    pass_rate: float
    critical_failures: int
    results: List[QueryResult]
    timestamp: str
    gate_passed: bool    # True if no critical failures and pass_rate above threshold


PASS_RATE_THRESHOLD = 0.90    # <90% pass rate fails the gate
CRITICAL_TOLERANCE = 0        # any critical failure fails the gate


def check_query(query: GoldenQuery, response: str, latency_ms: float) -> QueryResult:
    failures = []

    # Content checks
    for term in query.must_contain:
        if term.lower() not in response.lower():
            failures.append(f"missing_required_content: '{term}'")

    for term in query.must_not_contain:
        if term.lower() in response.lower():
            failures.append(f"contains_forbidden_content: '{term}'")

    # Length checks
    word_count = len(response.split())
    if word_count < query.min_length:
        failures.append(f"response_too_short: {word_count} words (min {query.min_length})")
    if word_count > query.max_length:
        failures.append(f"response_too_long: {word_count} words (max {query.max_length})")

    # Format check
    if query.expected_format == "json":
        try:
            json.loads(response)
        except json.JSONDecodeError:
            failures.append("format_not_json")
    elif query.expected_format == "list":
        if not any(marker in response for marker in ["- ", "* ", "1. ", "•"]):
            failures.append("format_not_list")

    return QueryResult(
        query_id=query.id,
        prompt=query.prompt,
        response=response,
        passed=len(failures) == 0,
        failures=failures,
        latency_ms=round(latency_ms, 2),
        severity=query.severity,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def run_harness(
    queries: List[GoldenQuery],
    model_fn: Callable[[str], str],
    model_version: str = "unknown",
) -> HarnessReport:
    """
    Runs all golden queries through model_fn and evaluates results.

    Args:
        queries: List of GoldenQuery objects
        model_fn: Function that takes a prompt string and returns a response string
        model_version: Label for this model/deployment (e.g. "claude-3-5-sonnet-20241022")
    """
    import hashlib
    run_id = hashlib.sha256(
        f"{model_version}{datetime.now().isoformat()}".encode()
    ).hexdigest()[:12]

    results = []
    for query in queries:
        start = time.time()
        try:
            response = model_fn(query.prompt)
        except Exception as e:
            response = f"[ERROR: {str(e)}]"
        latency_ms = (time.time() - start) * 1000

        result = check_query(query, response, latency_ms)
        results.append(result)

        status = "✅" if result.passed else "❌"
        print(f"  {status} [{query.severity.upper()}] {query.id} ({latency_ms:.0f}ms)")
        if result.failures:
            for f in result.failures:
                print(f"       └─ {f}")

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    critical_failures = sum(
        1 for r in results if not r.passed and r.severity == "critical"
    )
    pass_rate = passed / max(total, 1)
    gate_passed = pass_rate >= PASS_RATE_THRESHOLD and critical_failures <= CRITICAL_TOLERANCE

    return HarnessReport(
        run_id=run_id,
        model_version=model_version,
        total=total,
        passed=passed,
        failed=total - passed,
        pass_rate=round(pass_rate, 4),
        critical_failures=critical_failures,
        results=results,
        timestamp=datetime.now(timezone.utc).isoformat(),
        gate_passed=gate_passed,
    )


def load_golden_queries(path: str = "data/golden_queries.json") -> List[GoldenQuery]:
    with open(path) as f:
        raw = json.load(f)
    return [GoldenQuery(**q) for q in raw]


def save_report(report: HarnessReport, output_path: str = "outputs/harness_report.json") -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(asdict(report), f, indent=2)


def print_summary(report: HarnessReport) -> None:
    gate = "GATE PASSED ✅" if report.gate_passed else "GATE FAILED ❌"
    print(f"\n{'='*60}")
    print(f"  Eval Harness: {gate}")
    print(f"  Run ID:       {report.run_id}")
    print(f"  Model:        {report.model_version}")
    print(f"{'='*60}")
    print(f"  Total:        {report.total}")
    print(f"  Passed:       {report.passed} ({report.pass_rate:.1%})")
    print(f"  Failed:       {report.failed}")
    print(f"  Critical:     {report.critical_failures}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────
# Local RAG reliability harness (no API required)
# ─────────────────────────────────────────────────────────────


@dataclass
class RagEvalQuery:
    id: str
    query: str
    expect_abstain: bool
    severity: str
    description: str
    must_contain: List[str] = field(default_factory=list)
    min_confidence: float = 0.0
    min_faithfulness: float = 0.0


def load_rag_eval_queries(path: str = "data/rag_eval_queries.json") -> List[RagEvalQuery]:
    with open(path) as f:
        raw = json.load(f)
    return [RagEvalQuery(**q) for q in raw]


def check_rag_query(spec: RagEvalQuery, result) -> QueryResult:
    """Evaluate one RAG pipeline result against acceptance criteria."""
    failures: List[str] = []
    response = result.message

    if spec.expect_abstain and not result.abstained:
        failures.append("expected_abstention_but_answered")
    if not spec.expect_abstain and result.abstained:
        failures.append("unexpected_abstention")

    if not spec.expect_abstain:
        if result.confidence.score < spec.min_confidence:
            failures.append(
                f"confidence_below_floor: {result.confidence.score:.3f} < {spec.min_confidence}"
            )
        if result.faithfulness is None:
            failures.append("missing_faithfulness_score")
        elif result.faithfulness.score < spec.min_faithfulness:
            failures.append(
                f"faithfulness_below_floor: {result.faithfulness.score:.3f} < {spec.min_faithfulness}"
            )
        elif result.faithfulness.hallucination_risk == "high":
            failures.append("hallucination_risk_high")

        for term in spec.must_contain:
            if term.lower() not in response.lower():
                failures.append(f"missing_required_content: '{term}'")

    return QueryResult(
        query_id=spec.id,
        prompt=spec.query,
        response=response,
        passed=len(failures) == 0,
        failures=failures,
        latency_ms=0.0,
        severity=spec.severity,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def run_rag_harness(
    queries: Optional[List[RagEvalQuery]] = None,
    docs_path: str = "data/sample_docs.json",
    model_version: str = "rag-pipeline-v1",
) -> HarnessReport:
    """
    Runs the local grounded RAG pipeline against curated reliability cases.

    This is the default entry point for `python -m src.eval_harness` and does
    not require an external model API.
    """
    from src.pipeline import ReliabilityPipeline

    specs = queries if queries is not None else load_rag_eval_queries()
    pipeline = ReliabilityPipeline.from_docs(path=docs_path)

    import hashlib
    run_id = hashlib.sha256(
        f"{model_version}{datetime.now().isoformat()}".encode()
    ).hexdigest()[:12]

    results: List[QueryResult] = []
    for spec in specs:
        start = time.time()
        pipeline_result = pipeline.run(spec.query)
        latency_ms = (time.time() - start) * 1000
        result = check_rag_query(spec, pipeline_result)
        result.latency_ms = round(latency_ms, 2)
        results.append(result)

        status = "✅" if result.passed else "❌"
        print(f"  {status} [{spec.severity.upper()}] {spec.id} ({latency_ms:.0f}ms)")
        conf = pipeline_result.confidence.score
        faith = (
            pipeline_result.faithfulness.score
            if pipeline_result.faithfulness is not None
            else None
        )
        print(
            f"       abstain={pipeline_result.abstained} "
            f"confidence={conf:.3f} faithfulness={faith}"
        )
        if result.failures:
            for f in result.failures:
                print(f"       └─ {f}")

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    critical_failures = sum(
        1 for r in results if not r.passed and r.severity == "critical"
    )
    pass_rate = passed / max(total, 1)
    gate_passed = pass_rate >= PASS_RATE_THRESHOLD and critical_failures <= CRITICAL_TOLERANCE

    return HarnessReport(
        run_id=run_id,
        model_version=model_version,
        total=total,
        passed=passed,
        failed=total - passed,
        pass_rate=round(pass_rate, 4),
        critical_failures=critical_failures,
        results=results,
        timestamp=datetime.now(timezone.utc).isoformat(),
        gate_passed=gate_passed,
    )


def main() -> int:
    print("\n🔍 Running RAG reliability harness (local, no API)...\n")
    report = run_rag_harness()
    save_report(report, output_path="outputs/rag_harness_report.json")
    print_summary(report)
    return 0 if report.gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
