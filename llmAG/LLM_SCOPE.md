# LLM Scope and Acceptance Criteria

## Purpose
Define the behavioral scope and acceptance criteria for the RAG assistant used in this project.

## Core Principles
- Use only the provided context from ingested PDFs as the source of truth.
- Answers must be grounded in the retrieved context with inline citations.
- No guessing or use of prior knowledge outside the context.

## In-Scope Tasks
- Summarize a paper.
- Extract contributions.
- Compare methods (including across multiple papers when the question is general).
- Extract experimental setup and results.

## Out-of-Scope Tasks
- Identify limitations or future work.
- Generate related-work bullets from provided papers.

## Answer Format
- Inline citations after specific clauses: `[Title | Section]`.
- Each factual claim must have at least one citation.
- Long answers are allowed when needed.

## Unknown or Missing Context
- If the answer is not explicitly present in context:
  - Provide a brief reason.
  - Ask for permission to search online (no automatic search).
  - Example phrasing: "I do not know based on the provided context. Would you like me to find related papers online?"

## Cross-Paper Behavior
- If the question is general, answers may reference multiple papers.
- If the question targets a specific paper, do not mix in other papers.

## Prompt Templates
- Use two templates:
  - Answer template (normal RAG response with citations).
  - Insufficient-context template (brief reason + ask to search online).

## Acceptance Criteria
- Every factual claim is cited with `[Title | Section]`.
- Missing-context answers always include a brief reason and a permission question.
- No external knowledge is used unless the user explicitly approves online search.
