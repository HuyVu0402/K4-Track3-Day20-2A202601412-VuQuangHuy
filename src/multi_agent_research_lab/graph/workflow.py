"""LangGraph workflow skeleton."""

from dataclasses import dataclass

from multi_agent_research_lab.agents import (
    AnalystAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.state import ResearchState


@dataclass(frozen=True)
class CompiledWorkflow:
    """Small compiled workflow object used by the runner."""

    supervisor: SupervisorAgent
    workers: dict[str, BaseAgent]


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def __init__(
        self,
        supervisor: SupervisorAgent | None = None,
        researcher: ResearcherAgent | None = None,
        analyst: AnalystAgent | None = None,
        writer: WriterAgent | None = None,
    ) -> None:
        self.supervisor = supervisor or SupervisorAgent()
        self.researcher = researcher or ResearcherAgent()
        self.analyst = analyst or AnalystAgent()
        self.writer = writer or WriterAgent()
        self._compiled: CompiledWorkflow | None = None

    def build(self) -> object:
        """Create the executable workflow graph."""

        self._compiled = CompiledWorkflow(
            supervisor=self.supervisor,
            workers={
                "researcher": self.researcher,
                "analyst": self.analyst,
                "writer": self.writer,
            },
        )
        return self._compiled

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state."""

        compiled = self._compiled or self.build()
        if not isinstance(compiled, CompiledWorkflow):
            raise AgentExecutionError("Workflow build returned an invalid compiled object.")

        settings = get_settings()
        max_steps = settings.max_iterations + 1

        for _ in range(max_steps):
            state = compiled.supervisor.run(state)
            route = state.route_history[-1] if state.route_history else "done"
            if route == "done":
                state.add_trace_event("workflow.done", {"iteration": state.iteration})
                return state

            worker = compiled.workers.get(route)
            if worker is None:
                state.errors.append(f"Unknown workflow route: {route}")
                state.add_trace_event("workflow.invalid_route", {"route": route})
                return state

            try:
                state = worker.run(state)
            except Exception as exc:
                state.errors.append(f"{route} failed: {exc}")
                state.add_trace_event(
                    "workflow.worker_error",
                    {"route": route, "error": str(exc)},
                )
                return state

        state.errors.append("Workflow stopped after reaching max orchestration steps.")
        state.add_trace_event("workflow.max_steps", {"max_steps": max_steps})
        return state
