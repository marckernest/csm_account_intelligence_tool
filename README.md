# Account Intelligence Tool

Reads Fathom transcript files and produces structured account intelligence using a local LM Studio model — free, private, no rate limits, handles your full call library. Claude API available as an optional flag for higher-quality runs on curated samples.

> **Model-agnostic by design.** Defaults to Qwen3.5 9B via LM Studio running locally. Optional `--use-claude` flag switches to Anthropic's Claude API. This is a deliberate architectural decision, not a limitation — local inference is the primary path; cloud API is the upgrade path for smaller, high-value samples.

---

## Prerequisites

- **Python 3.10+**
- **LM Studio** (default) with Qwen3.5-9B (or compatible model) loaded at `http://localhost:1234/v1`
- **Anthropic API key** (optional, for `--use-claude` mode)

## Setup

```bash
pip install openai anthropic python-dotenv
```

For Claude mode, create a `.env` file:

```
ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

```
python transcript_analyst.py [options]
```

### Default — Local LM Studio

```bash
# Process all .md files in ./fathom_transcripts/
python transcript_analyst.py

# Custom input/output directories
python transcript_analyst.py --input-dir ./calls --output-dir ./analysis

# Include full transcript text (default: condensed)
python transcript_analyst.py --full-transcripts

# Build aggregate report across all calls
python transcript_analyst.py --aggregate
```

### Claude API — Optional Upgrade

```bash
# Use Claude instead of local LM Studio
python transcript_analyst.py --use-claude

# Use Claude with a specific model
python transcript_analyst.py --use-claude --claude-model claude-sonnet-4-20250514
```

### All Options

| Flag | Default | Description |
|---|---|---|
| `--input-dir` | `./fathom_transcripts` | Directory with Fathom `.md` files |
| `--output-dir` | `./output` | Directory for output JSON files |
| `--aggregate` | off | Build a single `aggregate.json` from all results |
| `--use-claude` | off | Use Claude API instead of local LM Studio |
| `--full-transcripts` | off | Include full transcript text in prompt |
| `--model` | `local-model` | Model name for LM Studio |
| `--claude-model` | `claude-sonnet-4-20250514` | Claude model name |
| `--lm-studio-url` | `http://localhost:1234/v1` | LM Studio API base URL |

---

## Model Tradeoffs

| | **Default (LM Studio)** | **--use-claude** |
|---|---|---|
| **Cost** | Free (your hardware) | Per-token API pricing |
| **Privacy** | Fully local, no data leaves your machine | Data sent to Anthropic |
| **Speed** | Depends on hardware (GPU/VRAM) | Fast, cloud-hosted |
| **Quality** | Good (Qwen3.5 9B) | Excellent (Claude Sonnet) |
| **Rate limits** | None | API rate limits apply |
| **Volume** | Full call library | Best for curated samples |

### When to use which

- **Local LM Studio** — Daily driver. Run against your entire Fathom export. No costs, no limits.
- **Claude API** — Deep-dive on 5-10 key calls where higher analytical quality justifies the cost.

---

## Input

Place Fathom-format `.md` files in the input directory. Each file should contain:

```markdown
# AccountName: Meeting Title

## Meeting Info

- **Date:** 2025-05-05 11:46 UTC

## Action Items

- [ ] Task description *(assigned to: Person)*

## Transcript

[00:00:00] {'display_name': 'Speaker'}: Utterance
```

## Output

Per-file JSON in the output directory:

```json
{
  "key_insights": ["..."],
  "risks": ["..."],
  "opportunities": ["..."],
  "sentiment": "positive",
  "action_items": ["..."],
  "recommended_next_steps": ["..."],
  "health_score": 7,
  "_file": "2025-05-05_Passion_Joe_and_Marck.md",
  "_account_name": "Passion",
  "_call_date": "2025-05-05 11:46 UTC",
  "_participants": ["Ben Batting", "Marck Ernest"]
}
```

With `--aggregate`, an additional `aggregate.json` is generated with `overall` stats and all call results.

---

## Project Structure

```
.
├── transcript_analyst.py    # Main script
├── fathom_transcripts/      # Input: Fathom .md files (gitignored)
├── output/                  # Output: per-file + aggregate JSON (gitignored)
├── .env                     # API keys (gitignored)
├── README.md
└── Project_Plan_AccountIntelligenceTool.md
```

## Portfolio

> Runs locally on Qwen3.5 9B via LM Studio — free, private, no rate limits, handles your full call library. Claude API available as an optional flag for higher-quality runs on curated samples.
