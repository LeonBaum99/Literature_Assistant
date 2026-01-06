from __future__ import annotations

"""RAG orchestration layer for the sandbox pipeline.

Responsibilities:
1) Retrieve relevant context (via a retriever interface),
2) Assemble prompts with bounded context size,
3) Invoke the LLM using the shared build factory.

Workflow:
run(question) -> parse optional slash command -> retrieve docs -> choose prompt template
-> format context -> invoke LLM chain -> return RagResponse with status flags.
"""

from dataclasses import dataclass
from typing import Iterable, List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from llmAG.llm import build_llm
from llmAG.llm_config import (
    ANSWER_SYSTEM_PROMPT,
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEBUG_SYSTEM_PROMPT,
    INSUFFICIENT_SYSTEM_PROMPT,
    MODE_A_SYSTEM_PROMPT,
    MODE_B_SYSTEM_PROMPT,
    MODE_C_SYSTEM_PROMPT,
)
from .retrievers import BaseRetriever



@dataclass
class RagResponse:
    answer: str
    sources: Optional[List[Document]]
    # Status flags let callers implement multi-turn flows (e.g., permission to search).
    status: str = "ok"
    needs_search: bool = False
    template: str = "answer"


class RagPipeline:
    """Minimal RAG pipeline: retrieve -> format context -> prompt -> LLM."""

    _COMMAND_ALIASES = {
        "/answer": "answer",
        "/mode_a": "mode_a",
        "/general": "mode_a",
        "/mode_b": "mode_b",
        "/paper": "mode_b",
        "/mode_c": "mode_c",
        "/draft": "mode_c",
        "/debug": "debug",
    }

    def __init__(self, retriever: BaseRetriever, model: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE, max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,) -> None:
        """Initialize the pipeline; called once by backend or notebook setup."""
        # Abstract retriever to allow for chroma replacement
        self._retriever = retriever
        # LLM factory wrapper to centralize model config
        self._llm = build_llm(model=model, temperature=temperature)
        # Context length guard: limits prompt size without tokenizers.
        self._max_context_chars = max_context_chars

        # Cached prompt template and chains to avoid re-creation on each run.
        self._prompts = {
            "answer": ChatPromptTemplate.from_messages(
                [
                    ("system", ANSWER_SYSTEM_PROMPT),
                    ("human", "Question: {question}\n\nContext:\n{context}"),
                ]
            ),
            "mode_a": ChatPromptTemplate.from_messages(
                [
                    ("system", MODE_A_SYSTEM_PROMPT),
                    ("human", "Question: {question}\n\nContext:\n{context}"),
                ]
            ),
            "mode_b": ChatPromptTemplate.from_messages(
                [
                    ("system", MODE_B_SYSTEM_PROMPT),
                    ("human", "Question: {question}\n\nContext:\n{context}"),
                ]
            ),
            "mode_c": ChatPromptTemplate.from_messages(
                [
                    ("system", MODE_C_SYSTEM_PROMPT),
                    ("human", "Question: {question}\n\nContext:\n{context}"),
                ]
            ),
            "insufficient": ChatPromptTemplate.from_messages(
                [
                    ("system", INSUFFICIENT_SYSTEM_PROMPT),
                    ("human", "Question: {question}"),
                ]
            ),
            "debug": ChatPromptTemplate.from_messages(
                [
                    ("system", DEBUG_SYSTEM_PROMPT),
                    ("human", "Question: {question}\n\nContext:\n{context}"),
                ]
            ),
        }
        # Chains are cached on the instance to keep invocation cheap.
        # ChatPromptTemplate | LLM produces a callable chain in LangChain.
        self._chains = {
            name: prompt | self._llm for name, prompt in self._prompts.items()
        }


    def _format_context(self, docs: Iterable[Document]) -> str:
        """
        Build a context string; called by _select_chain before LLM invocation.
        Experimental --> Bit of a Blackbox
        """
        # Assemble a readable context block, truncating to max size.
        parts: List[str] = []
        current_len = 0
        for doc in docs:
            header = self._format_header(doc)
            snippet = doc.page_content.strip()
            chunk = f"{header}\n{snippet}"
            if self._max_context_chars and current_len + len(chunk) > self._max_context_chars:
                # Trim the final chunk to fit the remaining space without splitting mid-word.
                remaining = self._max_context_chars - current_len
                if remaining <= 0:
                    break
                chunk = self._truncate_chunk(chunk, remaining)
                if not chunk:
                    break
                parts.append(chunk)
                current_len += len(chunk)
                break
            parts.append(chunk)
            current_len += len(chunk)
        return "\n\n".join(parts) if parts else "No relevant context found."

    @staticmethod
    def _format_header(doc: Document) -> str:
        """Format a citation header; used by _format_context for each Document."""
        # Mirror the metadata produced in the PDF pipeline. --> good for debugging
        title = doc.metadata.get("title", "Unknown") if doc.metadata else "Unknown"
        section = doc.metadata.get("section", "Unknown") if doc.metadata else "Unknown"
        return f"[{title} | {section}]"

    @staticmethod
    def _truncate_chunk(chunk: str, remaining: int) -> str:
        """
        Experimental --> Bit of a blackbox
        Trim the last chunk without breaking words; used only by _format_context.
        """
        if remaining <= 0:
            return ""
        if len(chunk) <= remaining:
            return chunk
        cutoff = chunk.rfind("\n", 0, remaining)
        if cutoff == -1:
            cutoff = chunk.rfind(" ", 0, remaining)
        if cutoff == -1:
            cutoff = remaining
        return chunk[:cutoff].rstrip()

    def run(self, question: str, k: int = 4, include_sources: bool = True) -> RagResponse:
        """
        Main entry point; called by API handlers or notebooks.
        Retrieve top-k chunks and feed them into the prompt template.
        Optional slash commands override template selection but do not skip retrieval.
        """
        command, cleaned_question = self._parse_command(question)
        docs = self._retriever.get_relevant_documents(cleaned_question, k=k)
        chain_key, payload = self._select_chain(cleaned_question, docs, command)
        result = self._chains[chain_key].invoke(payload)
        answer = result.content if hasattr(result, "content") else str(result)
        sources = docs if include_sources else None
        status = "insufficient" if chain_key == "insufficient" else "ok"
        needs_search = chain_key == "insufficient"
        return RagResponse(
            answer=answer,
            sources=sources,
            status=status,
            needs_search=needs_search,
            template=chain_key,
        )

    @classmethod
    def _parse_command(cls, question: str) -> tuple[Optional[str], str]:
        """Parse leading slash commands; called by run before retrieval."""
        stripped = question.strip()
        if not stripped.startswith("/"):
            return None, question
        command_token, *rest = stripped.split(maxsplit=1)
        command = cls._COMMAND_ALIASES.get(command_token.lower())
        if not command:
            return None, question
        remaining = rest[0] if rest else ""
        # Return the resolved command plus the question with the command stripped.
        return command, remaining.strip()

    def _select_chain(self,question: str,docs: List[Document],command: Optional[str],) -> tuple[str, dict]:
        """
        Choose a prompt template; called by run after retrieval.
        """
        if not docs:
            return "insufficient", {"question": question}
        if command == "debug":
            return "debug", {
                "question": question,
                "context": self._format_context(docs)
            }
        if command in {"mode_a", "mode_b", "mode_c", "answer"}:
            chain_key = command
        else:
            chain_key = "answer"
        return chain_key, {
            "question": question,
            "context": self._format_context(docs)
        }
