"""Search client abstraction for ResearcherAgent."""

import json
import re
from pathlib import Path
from typing import Any

from multi_agent_research_lab.core.schemas import SourceDocument

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CORPUS_DIR = _REPO_ROOT / "ai_agent_offline_research_corpus_v2" / "topics"
_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")


class SearchClient:
    """Provider-agnostic search client skeleton."""

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query.

        The default implementation is an offline lexical search over the bundled
        benchmark corpus. This keeps tests and demos reproducible without API keys.
        """

        if max_results < 1:
            return []

        query_terms = _tokenize(query)
        candidates = list(self._iter_corpus_documents())
        ranked = sorted(
            candidates,
            key=lambda item: self._score(item, query_terms),
            reverse=True,
        )
        return [
            document
            for document in ranked[:max_results]
            if self._score(document, query_terms) > 0
        ] or ranked[:max_results]

    def _iter_corpus_documents(self) -> list[SourceDocument]:
        documents: list[SourceDocument] = []
        if not _CORPUS_DIR.exists():
            return documents

        for path in sorted(_CORPUS_DIR.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            topic = data.get("topic", {})
            knowledge_base = data.get("knowledge_base", {})
            topic_name = str(topic.get("name", path.stem))

            for article in knowledge_base.get("knowledge_articles", []):
                documents.append(
                    _source_from_article(article=article, topic_name=topic_name, path=path)
                )

            for source in knowledge_base.get("source_documents", []):
                documents.append(
                    _source_from_document(source=source, topic_name=topic_name, path=path)
                )

        return documents

    def _score(self, document: SourceDocument, query_terms: set[str]) -> int:
        if not query_terms:
            return 0

        haystack = " ".join([document.title, document.snippet])
        document_terms = _tokenize(haystack)
        metadata_terms = _tokenize(" ".join(str(value) for value in document.metadata.values()))
        title_terms = _tokenize(document.title)

        overlap = len(query_terms & document_terms)
        metadata_overlap = len(query_terms & metadata_terms)
        title_overlap = len(query_terms & title_terms)
        return overlap + metadata_overlap + (2 * title_overlap)


def _source_from_article(article: dict[str, Any], topic_name: str, path: Path) -> SourceDocument:
    article_id = str(article.get("article_id", "article"))
    content = str(article.get("content", ""))
    return SourceDocument(
        title=str(article.get("title", article_id)),
        url=None,
        snippet=_shorten(content),
        metadata={
            "source_id": article_id,
            "source_type": "knowledge_article",
            "topic": topic_name,
            "corpus_file": path.name,
        },
    )


def _source_from_document(source: dict[str, Any], topic_name: str, path: Path) -> SourceDocument:
    document_id = str(source.get("document_id", source.get("source_id", "source")))
    content = _first_text(
        source,
        keys=("summary", "content", "abstract", "key_findings", "relevance_note"),
    )
    return SourceDocument(
        title=str(source.get("title", document_id)),
        url=_optional_string(source.get("url")),
        snippet=_shorten(content),
        metadata={
            "source_id": document_id,
            "source_type": str(source.get("document_class", "source_document")),
            "topic": topic_name,
            "corpus_file": path.name,
            "is_synthetic": bool(source.get("is_synthetic", False)),
        },
    )


def _first_text(source: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, list) and value:
            return " ".join(str(item) for item in value)
    return str(source)


def _optional_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _shorten(text: str, limit: int = 700) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _tokenize(text: str) -> set[str]:
    return {match.group(0).lower() for match in _WORD_RE.finditer(text)}
