"""
run.py

CLI entry point for the LLM reliability toolkit.

Usage:
  python run.py harness          Run golden query regression harness
  python run.py drift            Run drift check against stored baseline
  python run.py snapshot         Save current responses as new baseline
  python run.py report           Print latest report summaries

Requires ANTHROPIC_API_KEY in environment.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime, timezone

from src.eval_harness import run_harness, load_golden_queries, save_report as save_harness_report, print_summary as print_harness_summary
from src.drift_monitor import run_drift_check, save_report as save_drift_report, print_summary as print_drift_summary


BASELINE_PATH = "data/baseline_responses.json"
SNAPSHOT_PATH = "outputs/snapshot.json"


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
                messages=[{"role": "user", "content": prompt}]
            )
            return msg.content[0].text

        return call_claude, model

    except ImportError:
        print("⚠️  anthropic package not installed. Run: pip install anthropic")
        sys.exit(1)


def cmd_harness():
    print("\n🔍 Running eval harness...\n")
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
        json.dump({
            "model_version": model_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "responses": snapshot,
        }, f, indent=2)

    print(f"\n✅ Baseline saved to {BASELINE_PATH} ({len(snapshot)} queries)")


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
    for report_path in ["outputs/harness_report.json", "outputs/drift_report.json"]:
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
            print(f"  {report_path} — not found (run harness or drift first)\n")


if __name__ == "__main__":
    commands = {
        "harness": cmd_harness,
        "snapshot": cmd_snapshot,
        "drift": cmd_drift,
        "report": cmd_report,
    }

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(f"Usage: python run.py [{' | '.join(commands.keys())}]")
        sys.exit(1)

    result = commands[sys.argv[1]]()
    sys.exit(result or 0)
