from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState


def test_supervisor_routes_by_missing_state_fields() -> None:
    supervisor = SupervisorAgent()
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))

    assert supervisor.choose_route(state) == "researcher"

    state.sources = [
        SourceDocument(
            title="Source",
            snippet="Evidence about multi-agent systems.",
            metadata={"source_id": "A01"},
        )
    ]
    state.research_notes = "Research notes"
    assert supervisor.choose_route(state) == "analyst"

    state.analysis_notes = "Analysis notes"
    assert supervisor.choose_route(state) == "writer"

    state.final_answer = "Final answer [A01]"
    assert supervisor.choose_route(state) == "done"


def test_supervisor_run_records_route_and_trace() -> None:
    supervisor = SupervisorAgent()
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))

    result = supervisor.run(state)

    assert result is state
    assert state.route_history == ["researcher"]
    assert state.iteration == 1
    assert state.trace[-1]["name"] == "supervisor.route"
    assert state.trace[-1]["payload"]["route"] == "researcher"
