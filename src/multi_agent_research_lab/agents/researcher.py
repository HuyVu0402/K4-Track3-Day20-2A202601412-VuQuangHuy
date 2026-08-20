"""Researcher agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(
        self,
        search_client: SearchClient | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.search_client = search_client or SearchClient()
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""

        sources = self.search_client.search(
            query=state.request.query,
            max_results=state.request.max_sources,
        )
        state.sources = sources

        source_brief = "\n".join(
            f"- [{source.metadata.get('source_id', index)}] {source.title}: {source.snippet}"
            for index, source in enumerate(sources, start=1)
        )
        response = self.llm_client.complete(
            system_prompt=(
                "You are the researcher in a multi-agent research workflow. "
                "Create concise evidence notes. Do not write the final answer."
            ),
            user_prompt=(
                f"Question: {state.request.query}\n"
                f"Audience: {state.request.audience}\n"
                f"Sources:\n{source_brief}"
            ),
        )

        state.research_notes = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=response.content,
                metadata={
                    "source_count": len(sources),
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event(
            "researcher.run",
            {
                "source_count": len(sources),
                "source_ids": [source.metadata.get("source_id") for source in sources],
            },
        )
        return state
