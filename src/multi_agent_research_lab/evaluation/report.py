"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to markdown."""

    lines = [
        "# Benchmark Report",
        "",
        "## Configuration",
        "",
        "- Source mode: bundled offline corpus",
        "- Cost: provider-reported cost when available; otherwise left blank",
        "- Quality: lightweight 0-10 heuristic from output completeness, sources, notes, "
        "citations, and errors",
        "",
        "## Results",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        citation = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure = "" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} "
            f"| {citation} | {failure} | {_escape_cell(item.notes)} |"
        )
    lines.extend(
        [
            "",
            "## Analysis",
            "",
            _analysis(metrics),
            "",
            "## Limitations",
            "",
            "- Offline fallback responses are deterministic and useful for plumbing checks, "
            "not final human-quality evaluation.",
            "- Citation coverage only checks whether source identifiers appear in the final "
            "answer; it does not prove entailment.",
            "- Run a real model and peer rubric before using this report as submission evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def _analysis(metrics: list[BenchmarkMetrics]) -> str:
    if not metrics:
        return "No benchmark runs were recorded."

    successful = [item for item in metrics if item.failure_rate in (None, 0)]
    fastest = min(metrics, key=lambda item: item.latency_seconds)
    best_quality = max(metrics, key=lambda item: item.quality_score or 0)
    return (
        f"- Fastest run: `{fastest.run_name}` at {fastest.latency_seconds:.2f}s.\n"
        f"- Highest heuristic quality: `{best_quality.run_name}` "
        f"({(best_quality.quality_score or 0):.1f}/10).\n"
        f"- Successful runs: {len(successful)}/{len(metrics)}."
    )


def _escape_cell(value: str) -> str:
    return value.replace("|", "\\|")
