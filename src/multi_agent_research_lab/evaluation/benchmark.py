"""Benchmark skeleton for single-agent vs multi-agent."""

from collections.abc import Callable
from statistics import mean
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency and derive lightweight metrics from the returned state."""

    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started
    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=_total_cost(state),
        quality_score=_quality_score(state),
        citation_coverage=_citation_coverage(state),
        failure_rate=1.0 if state.errors else 0.0,
        notes=_notes(state),
    )
    return state, metrics


def summarize_metrics(run_name: str, metrics: list[BenchmarkMetrics]) -> BenchmarkMetrics:
    """Aggregate repeated benchmark runs for one approach."""

    if not metrics:
        return BenchmarkMetrics(
            run_name=run_name,
            latency_seconds=0.0,
            failure_rate=1.0,
            notes="No benchmark runs were recorded.",
        )

    costs = [item.estimated_cost_usd for item in metrics if item.estimated_cost_usd is not None]
    qualities = [item.quality_score for item in metrics if item.quality_score is not None]
    citations = [item.citation_coverage for item in metrics if item.citation_coverage is not None]
    failures = [item.failure_rate for item in metrics if item.failure_rate is not None]

    return BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=mean(item.latency_seconds for item in metrics),
        estimated_cost_usd=None if not costs else mean(costs),
        quality_score=None if not qualities else mean(qualities),
        citation_coverage=None if not citations else mean(citations),
        failure_rate=None if not failures else mean(failures),
        notes=f"Mean of {len(metrics)} run(s).",
    )


def _total_cost(state: ResearchState) -> float | None:
    costs = [
        result.metadata.get("cost_usd")
        for result in state.agent_results
        if result.metadata.get("cost_usd") is not None
    ]
    if not costs:
        return None
    return float(sum(float(cost) for cost in costs))


def _quality_score(state: ResearchState) -> float:
    score = 0.0
    if state.final_answer:
        score += 3.0
        if len(state.final_answer.split()) >= 40:
            score += 1.0
    if state.sources:
        score += min(2.0, len(state.sources) / 2.5)
    if state.research_notes:
        score += 1.0
    if state.analysis_notes:
        score += 1.0
    if _citation_coverage(state) > 0:
        score += 1.0
    if state.errors:
        score -= 2.0
    return max(0.0, min(10.0, score))


def _citation_coverage(state: ResearchState) -> float:
    if not state.sources or not state.final_answer:
        return 0.0

    cited = 0
    for source in state.sources:
        source_id = source.metadata.get("source_id")
        if source_id and f"[{source_id}]" in state.final_answer:
            cited += 1
    return cited / len(state.sources)


def _notes(state: ResearchState) -> str:
    token_total = sum(
        int(result.metadata.get("input_tokens") or 0)
        + int(result.metadata.get("output_tokens") or 0)
        for result in state.agent_results
    )
    parts = [
        f"sources={len(state.sources)}",
        f"tokens={token_total}" if token_total else "tokens=unreported",
    ]
    if state.route_history:
        parts.append("route=" + "->".join(state.route_history))
    if state.errors:
        parts.append("errors=" + "; ".join(state.errors))
    return ", ".join(parts)
