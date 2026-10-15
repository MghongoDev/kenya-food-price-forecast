# Domain docs

## Layout: single-context

- `GLOSSARY.md` at the repo root — canonical glossary of this project's domain terms.
- `docs/adr/` — Architecture Decision Records, numbered `NNNN-title.md`.

## Consumer rules for agents

- Before working on any domain-flavored area, read the relevant `GLOSSARY.md` entries first.
- When a new domain term appears, record it in `GLOSSARY.md` instead of re-defining it inline.
- Check `docs/adr/` before making a non-obvious design decision; if no ADR covers it, write one (context, decision, consequences).
- Don't silently contradict an ADR — supersede it with a new ADR.
