"""Offline evaluation harness for the copilot.

Runs a golden question set and scores routing accuracy, answer checks
(expected rows/columns/phrases), citation coverage, and latency. Run it after
any prompt, router, or metric change; CI fails if accuracy drops below the bar.

    python -m src.evaluate             # print a scorecard, write reports/eval_results.md
    python -m src.evaluate --min 0.9   # exit 1 when overall pass rate < 90%
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from src.agent import answer_question
from src.config import PROJECT_ROOT

GOLDEN_PATH = PROJECT_ROOT / "evals" / "golden_questions.json"
RESULTS_PATH = PROJECT_ROOT / "reports" / "eval_results.md"


@dataclass
class CaseResult:
    id: str
    route_ok: bool
    checks_ok: bool
    cited: bool
    latency_ms: int
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.route_ok and self.checks_ok and self.cited


def run_case(case: dict) -> CaseResult:
    result = answer_question(case["question"])
    failures: list[str] = []

    route_ok = result.route == case["expected_route"]
    if not route_ok:
        failures.append(f"route={result.route}, expected {case['expected_route']}")

    if "expected_rows" in case and len(result.rows) != case["expected_rows"]:
        failures.append(f"rows={len(result.rows)}, expected {case['expected_rows']}")
    for column in case.get("expected_columns", []):
        if column not in result.columns:
            failures.append(f"missing column {column}")
    for phrase in case.get("must_contain", []):
        if phrase not in result.answer:
            failures.append(f"answer missing '{phrase}'")
    checks_ok = not [f for f in failures if not f.startswith("route=")]

    cited = "**Sources:**" in result.answer[-500:]
    if not cited:
        failures.append("no Sources section")

    return CaseResult(case["id"], route_ok, checks_ok, cited, result.latency_ms, failures)


def evaluate(path: Path = GOLDEN_PATH) -> tuple[list[CaseResult], dict[str, float]]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    results = [run_case(case) for case in cases]
    n = len(results)
    latencies = [r.latency_ms for r in results]
    scores = {
        "cases": n,
        "pass_rate": sum(r.passed for r in results) / n,
        "route_accuracy": sum(r.route_ok for r in results) / n,
        "answer_check_rate": sum(r.checks_ok for r in results) / n,
        "citation_rate": sum(r.cited for r in results) / n,
        "p50_latency_ms": statistics.median(latencies),
        "max_latency_ms": max(latencies),
    }
    return results, scores


def render(results: list[CaseResult], scores: dict[str, float]) -> str:
    lines = [
        "# Copilot evaluation scorecard",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Cases | {scores['cases']} |",
        f"| Overall pass rate | {scores['pass_rate']:.0%} |",
        f"| Routing accuracy | {scores['route_accuracy']:.0%} |",
        f"| Answer checks | {scores['answer_check_rate']:.0%} |",
        f"| Citation coverage | {scores['citation_rate']:.0%} |",
        f"| p50 latency | {scores['p50_latency_ms']:.0f} ms |",
        "",
        "| Case | Result | Latency | Notes |",
        "| --- | --- | --- | --- |",
    ]
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"| {r.id} | {status} | {r.latency_ms} ms | {'; '.join(r.failures) or '-'} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the copilot on the golden set.")
    parser.add_argument("--min", type=float, default=0.0, help="Minimum pass rate (0-1).")
    args = parser.parse_args()

    results, scores = evaluate()
    report = render(results, scores)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(report, encoding="utf-8")
    print(report)
    if scores["pass_rate"] < args.min:
        raise SystemExit(f"Pass rate {scores['pass_rate']:.0%} is below {args.min:.0%}.")


if __name__ == "__main__":
    main()
