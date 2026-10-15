# ADR 0004 - Deploy on Streamlit Community Cloud

## Status

Accepted

## Context

The MVP ends with a dashboard a stranger can open. Options: run it locally, deploy to Streamlit Community Cloud, or pay for a host. The repo is already on GitHub, which is what the free deploy consumes. The app must train nothing at load time, so precomputed forecast and backtest files drive it.

## Decision

Deploy at Phase 5 to Streamlit Community Cloud. The app, its precomputed files, and a pinned `requirements.txt` deploy from this repo. The final connect-GitHub step runs in the owner's browser.

## Consequences

Free, public, and no infrastructure to maintain. Community Cloud sets its own resource limits, so the app must stay light. If limits bite, the fallback is local running while a paid host is chosen.
