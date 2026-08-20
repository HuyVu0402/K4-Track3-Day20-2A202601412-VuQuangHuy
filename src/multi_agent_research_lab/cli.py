"""Command-line entrypoint for the lab starter."""

from time import perf_counter
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline."""

    _init()
    request = _parse_query(query)
    started = perf_counter()
    state = ResearchState(request=request)

    sources = SearchClient().search(request.query, max_results=request.max_sources)
    state.sources = sources
    source_brief = "\n".join(
        f"- [{source.metadata.get('source_id', index)}] {source.title}: {source.snippet}"
        for index, source in enumerate(sources, start=1)
    )
    response = LLMClient().complete(
        system_prompt=(
            "You are a single-agent research baseline. Answer the question directly "
            "using only the provided sources, and cite source identifiers in square brackets."
        ),
        user_prompt=(
            f"Question: {request.query}\n"
            f"Audience: {request.audience}\n"
            f"Sources:\n{source_brief or 'No sources available.'}"
        ),
    )
    state.final_answer = response.content
    latency = perf_counter() - started
    state.add_trace_event(
        "baseline.run",
        {
            "source_count": len(sources),
            "latency_seconds": latency,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost_usd": response.cost_usd,
        },
    )
    console.print(Panel.fit(state.final_answer, title="Single-Agent Baseline"))
    console.print(f"Sources: {len(sources)} | Latency: {latency:.2f}s")


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow."""

    _init()
    state = ResearchState(request=_parse_query(query))
    workflow = MultiAgentWorkflow()
    result = workflow.run(state)
    console.print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
