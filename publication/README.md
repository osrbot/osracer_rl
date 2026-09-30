# Public deliverables

This directory groups material intended for publishing or presentation:

- `assets/`: the reviewed media shared by the website, slide deck, and video tooling.
- `site/`: the dependency-free static evidence-site builder. `_site/` is generated.
- `presentation/`: the editable deck, speaker notes, media manifest, and video tooling.

Core code remains under `src/`; optional native adapters remain under `native/`. Published material may describe them but must not become a runtime dependency. Presentation-specific briefs and media stay here rather than in `docs/`.
