"""Writer agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""

        source_brief = "\n".join(
            (
                f"- [{source.metadata.get('source_id', index)}] {source.title}: "
                f"{source.snippet}"
            )
            for index, source in enumerate(state.sources, start=1)
        )
        response = self.llm_client.complete(
            system_prompt=(
                "You are the writer in a multi-agent research workflow. "
                "Write the final answer using only the provided notes and sources. "
                "Cite source identifiers in square brackets."
            ),
            user_prompt=(
                f"Question: {state.request.query}\n"
                f"Audience: {state.request.audience}\n\n"
                f"Research notes:\n{state.research_notes or 'No research notes available.'}\n\n"
                f"Analysis notes:\n{state.analysis_notes or 'No analysis notes available.'}\n\n"
                f"Sources:\n{source_brief or 'No sources available.'}"
            ),
        )

        state.final_answer = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
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
            "writer.run",
            {
                "source_count": len(state.sources),
                "has_research_notes": bool(state.research_notes),
                "has_analysis_notes": bool(state.analysis_notes),
            },
        )
        return state
