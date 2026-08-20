"""Analyst agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""

        source_brief = "\n".join(
            f"- [{source.metadata.get('source_id', index)}] {source.title}"
            for index, source in enumerate(state.sources, start=1)
        )
        response = self.llm_client.complete(
            system_prompt=(
                "You are the analyst in a multi-agent research workflow. "
                "Extract claims, compare evidence, and identify limits."
            ),
            user_prompt=(
                f"Question: {state.request.query}\n"
                f"Research notes:\n{state.research_notes or 'No research notes available.'}\n"
                f"Sources:\n{source_brief or 'No sources available.'}"
            ),
        )

        state.analysis_notes = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=response.content,
                metadata={
                    "source_count": len(state.sources),
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event(
            "analyst.run",
            {
                "source_count": len(state.sources),
                "has_research_notes": bool(state.research_notes),
            },
        )
        return state
