# Production Readiness

## Status

Planned

## Minimum Readiness Checklist

- Runtime and dependency versions pinned and documented.
- Secrets are injected from environment or secret manager, never hardcoded.
- Health checks and startup validation are present.
- Logging and error handling avoid leaking internals.
- Basic test and lint verification are automated.
- Deployment and rollback steps are documented.
- Monitoring and alerting ownership is defined.

## Verification Commands

```bash
python -m pytest -q
python -m ruff check .
```

## Open Risks

- To be completed during environment-specific hardening.
