# Project 2: Account Intelligence Tool (GitHub)

**Goal:** A Python script that reads Fathom transcript files and produces structured account intelligence using a local LM Studio model. Model-agnostic: local by default, Claude API optional for higher-quality runs on smaller samples.

**Scope boundary:** Single file. Reads from `./fathom_transcripts/` directory (fathom_downloader output). No web UI, no database. Local LM Studio endpoint by default. Runs free at any volume.

**Total estimated time:** ~4 hours

| Phase | Owner | What | Output |
|---|---|---|---|
| 1. Spec | LLM | Define input/output contract, prompt design, model config approach | One-page spec for Marck to approve |
| 2. Build | LLM | Write `transcript_analyst.py` | Working script |
| 3. Test | Marck | Point at LM Studio, run against a handful of real transcripts | Test feedback |
| 4. Revise | LLM | Fix anything broken or off | Final working script |
| 5. README | LLM | Document both modes, model tradeoffs, sample vs. full-library guidance | `README.md` |
| 6. Publish | Marck | Push both tools to GitHub as a single repo, make public | Live GitHub URL |

**Constraints:**
- Default endpoint: `http://localhost:1234/v1` (LM Studio OpenAI-compatible)
- Default model: Qwen3.5 9B, 8k context window
- Optional flag: `--use-claude` for Claude API runs on curated samples
- Input: Fathom summaries + action items by default. Full transcripts available as optional flag for deep-dive runs.
- No input trimming needed — 8k context comfortably fits a full Fathom summary, action items, and structured prompt in one shot
- README explicitly frames the model-agnostic design as a decision, not a limitation
- fathom_downloader Fathom API key swap (personal account) is a prerequisite — do this first

**Portfolio story:** "Runs locally on Qwen3.5 9B via LM Studio — free, private, no rate limits, handles your full call library. Claude API available as an optional flag for higher-quality runs on curated samples."
