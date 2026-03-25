# archive-enterprise-analytics — Copilot Instructions

Repository stack: Python analytics/application project.

## Mandatory Workflow

1. Plan before editing.
2. Keep changes scoped to root cause.
3. Verify before completion.

## Verification Gate

`ash
python -m pytest -q
python -m ruff check .
python -m pyright
`

Use only the commands available in this repo environment.

## Security Rules

- No fallback secrets in code.
- No raw exception leakage to users.
- Keep logging structured and non-sensitive.
