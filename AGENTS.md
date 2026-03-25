# AGENTS.md

Operational guide for AI agents in archive-enterprise-analytics.

## Completion Checklist

1. Implement minimal, scoped changes.
2. Run verification commands:
   - python -m pytest -q
   - python -m ruff check .
   - python -m pyright
3. Resolve diagnostics in changed files.
4. Summarize change, risk, and validation.

## Guardrails

- Do not introduce secret defaults.
- Do not leak internal errors in client responses.
- Prefer root-cause fixes over broad refactors.
