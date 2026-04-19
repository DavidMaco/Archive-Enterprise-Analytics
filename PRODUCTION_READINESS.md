# Production Readiness

## Status

**Production-Ready (Candidate)** — all quality gates pass; deployment to Streamlit Cloud is pending a go/no-go decision.

## Minimum Readiness Checklist

- [x] **Runtime and dependency versions pinned** — `runtime.txt` declares `python-3.11`; every package pinned in `requirements.txt` (pandas 2.3.3, scikit-learn 1.8.0, streamlit 1.54.0, etc.).
- [x] **Secrets injected from environment, never hardcoded** — `src/archive_analytics/config.py` reads `ARCHIVE_ANALYTICS_RAW_DIR` from environment. Secret values documented in `.streamlit/secrets.toml.example`; the real `secrets.toml` is `.gitignore`d.
- [x] **Health checks and startup validation present** — `src/archive_analytics/data.py::processed_assets_ready()` validates all required parquet/model files exist at startup; each dashboard page calls `require_dashboard_assets()` and shows a clear error if assets are missing.
- [x] **Logging and error handling avoid leaking internals** — all modules use `logging.getLogger(__name__)`; no stack traces are exposed to the Streamlit UI (caught and surfaced as user-facing messages).
- [x] **Test and lint verification automated** — GitHub Actions CI at `.github/workflows/ci.yml` runs on every push: ruff → mypy → pytest (Python 3.11 + 3.12 matrix).
- [x] **All three local quality gates pass**:
  - `python -m pytest -q` → 39/39 passing
  - `python -m ruff check .` → no issues found
  - `python -m pyright` → 0 errors, 95 warnings (all pandas/sklearn stubs-friction)
- [x] **Deployment steps documented** — see README.md "Deploying to Streamlit Cloud" section.
- [x] **Processed data and model artefacts committed** — `data/processed/` and `data/models/` are committed so Streamlit Cloud can serve the dashboard without running the full pipeline.
- [ ] **Monitoring and alerting ownership defined** — *Gap*. Streamlit Cloud provides no native alerting. Mitigation: set up a UptimeRobot or similar external HTTP check against the app URL; assign on-call ownership before launch.

## Verification Commands

```bash
# All three gates must exit 0
python -m pytest -q
python -m ruff check .
python -m pyright
```

## Deployment Procedure

1. Fork/push this branch to GitHub (`main`).
2. Connect the repo to Streamlit Cloud (Settings → Secrets → paste contents of `.streamlit/secrets.toml.example` with real values).
3. Set `ARCHIVE_ANALYTICS_RAW_DIR` in Streamlit Cloud secrets if raw data re-ingestion is needed.
4. Streamlit Cloud will install `requirements.txt` and run `streamlit_app.py` automatically.
5. Verify the landing page loads and the six dashboard pages render without errors.

## Rollback Procedure

```bash
# Identify the previous stable commit
git log --oneline -10

# Revert to it
git revert HEAD  # or: git revert <commit-sha>
git push origin main
# Streamlit Cloud auto-redeploys on push
```

## Open Risks

| Risk | Severity | Mitigation |
|---|---|---|
| No monitoring/alerting defined | Medium | Add UptimeRobot HTTP check; assign on-call owner before launch |
| Streamlit Cloud cold-start latency (~30 s) | Low | Pre-warm via scheduled ping or accept documented limitation |
| Stubs-friction pyright warnings (95) | Info | All suppressed with targeted `# type: ignore` or config overrides; no runtime impact |
