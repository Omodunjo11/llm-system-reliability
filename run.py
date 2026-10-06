"""
run.py

CLI entry point for the LLM reliability toolkit.

Usage:
  python run.py rag [query]      Run grounded RAG pipeline (demo / interactive)
  python run.py eval             Run local RAG reliability harness (no API)
  python run.py audit            Run adversarial break harness (no API)
  python run.py harness          Run golden query regression harness (LLM / mock)
  python run.py drift            Run drift check against stored baseline
  python run.py snapshot         Save current responses as new baseline
  python run.py report           Print latest report summaries

LLM harness/drift/snapshot use ANTHROPIC_API_KEY when set; otherwise mock mode.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.drift_monitor import (
    print_summary as print_drift_summary,
    run_drift_check,
    save_report as save_drift_report,
)
from src.eval_harness import (
    load_golden_queries,
    print_summary as print_harness_summary,
    run_adversarial_harness,
    run_harness,
    run_rag_harness,
    save_report as save_harness_report,
)
from src.pipeline import ReliabilityPipeline


BASELINE_PATH = "data/baseline_responses.json"


def get_model_fn():
    """
    Returns a function that calls the Claude API.
    Falls back to a mock function if no API key is set.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("⚠️  No ANTHROPIC_API_KEY found. Using mock responses for demo.")

        def mock_fn(prompt: str) -> str:
            return f"[MOCK RESPONSE] This is a placeholder response for: {prompt[:80]}..."

        return mock_fn, "mock-v0"

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        model = "claude-3-5-sonnet-20241022"

        def call_claude(prompt: str) -> str:
            msg = client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text

        return call_claude, model

    except ImportError:
        print("⚠️  anthropic package not installed. Run: pip install anthropic")
        sys.exit(1)


def cmd_rag(query: str | None = None):
    if not query:
        query = input("Enter your query: ").strip()
    if not query:
        print("No query provided.")
        return 1

    pipeline = ReliabilityPipeline.from_docs()
    result = pipeline.run(query)

    print("\n=== RETRIEVAL ===")
    if not result.retrieved:
        print("  (no documents above score floor)")
    for doc in result.retrieved:
        print(f"  [{doc.id}] score={doc.score:.3f} terms={doc.matched_terms}")
        print(f"    {doc.text[:140]}{'...' if len(doc.text) > 140 else ''}")

    print("\n=== CONFIDENCE ===")
    c = result.confidence
    print(f"  score={c.score:.3f} hard_abstain={c.hard_abstain}")
    print(
        f"  strength={c.retrieval_strength:.3f} margin={c.score_margin:.3f} "
        f"coverage={c.term_coverage:.3f} idf_coverage={c.idf_coverage:.3f} "
        f"density={c.evidence_density:.3f}"
    )
    if c.missing_rare_terms:
        print(f"  missing_rare={c.missing_rare_terms}")
    print(f"  reasons={c.reasons}")

    if result.abstained:
        print("\n=== RESULT ===")
        print(result.message)
        if result.faithfulness:
            print("\n=== FAITHFULNESS (pre-abstain generation) ===")
            print(json.dumps(result.faithfulness.as_dict(), indent=2))
        return 0

    print("\n=== ANSWER ===")
    print(result.message)

    print("\n=== FAITHFULNESS ===")
    print(json.dumps(result.faithfulness.as_dict(), indent=2))
    return 0


def cmd_eval():
    print("\n🔍 Running local RAG reliability harness...\n")
    report = run_rag_harness()
    save_harness_report(report, output_path="outputs/rag_harness_report.json")
    print_harness_summary(report)
    return 0 if report.gate_passed else 1


def cmd_audit():
    print("\n💥 Running adversarial break harness...\n")
    report = run_adversarial_harness()
    save_harness_report(report, output_path="outputs/adversarial_harness_report.json")
    print_harness_summary(report)
    return 0 if report.gate_passed else 1


def cmd_harness():
    print("\n🔍 Running LLM golden-query harness...\n")
    queries = load_golden_queries()
    model_fn, model_version = get_model_fn()
    report = run_harness(queries, model_fn, model_version)
    save_harness_report(report)
    print_harness_summary(report)
    return 0 if report.gate_passed else 1


def cmd_snapshot():
    print("\n📸 Saving response snapshot as baseline...\n")
    queries = load_golden_queries()
    model_fn, model_version = get_model_fn()

    snapshot = {}
    for q in queries:
        print(f"  Querying: {q.id}")
        snapshot[q.id] = model_fn(q.prompt)

    Path(BASELINE_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_PATH, "w") as f:
        json.dump(
            {
                "model_version": model_version,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "responses": snapshot,
            },
            f,
            indent=2,
        )

    print(f"\n✅ Baseline saved to {BASELINE_PATH} ({len(snapshot)} queries)")
    return 0


def cmd_drift():
    if not Path(BASELINE_PATH).exists():
        print(f"❌ No baseline found at {BASELINE_PATH}. Run: python run.py snapshot")
        sys.exit(1)

    print("\n📊 Running drift check...\n")

    with open(BASELINE_PATH) as f:
        baseline_data = json.load(f)

    queries = load_golden_queries()
    model_fn, current_version = get_model_fn()

    current = {}
    for q in queries:
        print(f"  Querying: {q.id}")
        current[q.id] = model_fn(q.prompt)

    report = run_drift_check(
        baseline=baseline_data["responses"],
        current=current,
        baseline_version=baseline_data.get("model_version", "baseline"),
        current_version=current_version,
    )
    save_drift_report(report)
    print_drift_summary(report)
    return 0 if report.passed else 1


def cmd_report():
    print("\n📋 Latest reports:\n")
    for report_path in [
        "outputs/rag_harness_report.json",
        "outputs/adversarial_harness_report.json",
        "outputs/harness_report.json",
        "outputs/drift_report.json",
    ]:
        if Path(report_path).exists():
            with open(report_path) as f:
                data = json.load(f)
            print(f"  {report_path}")
            print(f"    Timestamp: {data.get('timestamp', 'unknown')}")
            if "gate_passed" in data:
                status = "PASS ✅" if data["gate_passed"] else "FAIL ❌"
                print(f"    Harness:   {status} ({data.get('pass_rate', 0):.1%} pass rate)")
            if "passed" in data and "drift_rate" in data:
                status = "PASS ✅" if data["passed"] else "FAIL ❌"
                print(f"    Drift:     {status} ({data.get('drift_rate', 0):.1%} drift rate)")
            print()
        else:
            print(f"  {report_path} — not found (run eval, harness, or drift first)\n")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python run.py [rag | eval | audit | harness | snapshot | drift | report] "
            "[optional query for rag]"
        )
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "rag":
        query = " ".join(sys.argv[2:]).strip() or None
        sys.exit(cmd_rag(query) or 0)
    if cmd == "eval":
        sys.exit(cmd_eval() or 0)
    if cmd == "audit":
        sys.exit(cmd_audit() or 0)
    if cmd == "harness":
        sys.exit(cmd_harness() or 0)
    if cmd == "snapshot":
        sys.exit(cmd_snapshot() or 0)
    if cmd == "drift":
        sys.exit(cmd_drift() or 0)
    if cmd == "report":
        sys.exit(cmd_report() or 0)

    print(f"Unknown command: {cmd}")
    sys.exit(1)
