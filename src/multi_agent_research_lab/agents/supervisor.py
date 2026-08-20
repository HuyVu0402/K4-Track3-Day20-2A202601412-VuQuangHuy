"""Supervisor / router skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route."""

        route = self.choose_route(state)
        state.record_route(route)
        state.add_trace_event(
            "supervisor.route",
            {
                "route": route,
                "iteration": state.iteration,
                "has_sources": bool(state.sources),
                "has_research_notes": bool(state.research_notes),
                "has_analysis_notes": bool(state.analysis_notes),
                "has_final_answer": bool(state.final_answer),
                "error_count": len(state.errors),
            },
        )
        return state

    def choose_route(self, state: ResearchState) -> str:
        """Choose the next worker from the current shared state."""

        settings = get_settings()
        if state.iteration >= settings.max_iterations:
            return "done"
        if state.errors and state.iteration >= max(1, settings.max_iterations - 1):
            return "done"
        if not state.sources or not state.research_notes:
            return "researcher"
        if not state.analysis_notes:
            return "analyst"
        if not state.final_answer:
            return "writer"
        return "done"
