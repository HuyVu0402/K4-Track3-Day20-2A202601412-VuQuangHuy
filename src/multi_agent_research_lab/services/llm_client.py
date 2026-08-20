"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from dataclasses import dataclass
from time import sleep

from multi_agent_research_lab.core.config import get_settings


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client skeleton."""

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion.

        Uses OpenAI when configured, then falls back to a deterministic offline
        completion so the lab can run without network credentials.
        """

        settings = get_settings()
        api_key = _configured_api_key(settings.openai_api_key)
        if api_key:
            response = self._complete_openai(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=settings.openai_model,
                api_key=api_key,
                timeout_seconds=settings.timeout_seconds,
            )
            if response is not None:
                return response

        return self._complete_offline(system_prompt=system_prompt, user_prompt=user_prompt)

    def _complete_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        api_key: str,
        timeout_seconds: int,
    ) -> LLMResponse | None:
        try:
            from openai import OpenAI
        except ImportError:
            return None

        client = OpenAI(api_key=api_key, timeout=timeout_seconds, max_retries=0)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.2,
                )
                usage = completion.usage
                return LLMResponse(
                    content=completion.choices[0].message.content or "",
                    input_tokens=None if usage is None else usage.prompt_tokens,
                    output_tokens=None if usage is None else usage.completion_tokens,
                )
            except Exception as exc:  # pragma: no cover - provider/network dependent
                last_error = exc
                sleep(0.5 * (attempt + 1))

        return LLMResponse(
            content=(
                "LLM provider call failed after retries. "
                f"Falling back to offline mode. Error: {last_error}"
            )
        )

    def _complete_offline(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        prompt = " ".join(part.strip() for part in (system_prompt, user_prompt) if part.strip())
        words = prompt.split()
        excerpt = " ".join(words[:80])
        content = (
            "Offline LLM response: use the available sources to answer conditionally, "
            "cite evidence, and compare quality, cost, latency, and failure modes. "
            f"Prompt excerpt: {excerpt}"
        )
        return LLMResponse(
            content=content,
            input_tokens=len(words),
            output_tokens=len(content.split()),
            cost_usd=None,
        )


def _configured_api_key(value: str | None) -> str | None:
    if value is None:
        return None

    stripped = value.strip()
    if not stripped or stripped.lower() in {"offline", "none", "null", "dummy", "test"}:
        return None
    return stripped
