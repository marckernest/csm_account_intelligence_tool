#!/usr/bin/env python3
"""transcript_analyst.py — Account Intelligence Tool

Reads Fathom transcript .md files and produces structured account intelligence
using a local LM Studio model (default) or optionally the Claude API.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


def parse_transcript(path):
    text = path.read_text(encoding="utf-8")

    title = ""
    m = re.search(r"^#\s+(.+)", text, re.MULTILINE)
    if m:
        title = m.group(1).strip()

    account_name = title.split(":")[0].strip() if ":" in title else title

    date = ""
    m = re.search(r"\*\*Date:\*\*\s*(.+)", text)
    if m:
        date = m.group(1).strip()

    duration = ""
    m = re.search(r"\*\*Duration:\*\*\s*(.+)", text)
    if m:
        duration = m.group(1).strip()

    action_items = []
    for m in re.finditer(
        r"- \[([ x])\]\s+(.+?)(?:\s+\*\(assigned to:\s*(.+?)\)\*)?$",
        text,
        re.MULTILINE,
    ):
        action_items.append(
            {
                "done": m.group(1) == "x",
                "text": m.group(2).strip(),
                "assigned_to": m.group(3).strip() if m.group(3) else "",
            }
        )

    transcript_start = text.find("## Transcript\n")
    if transcript_start == -1:
        transcript_start = text.find("## Transcript")
    if transcript_start == -1:
        transcript_text = ""
    else:
        transcript_text = text[transcript_start:].strip()

    participants = set()
    for m in re.finditer(
        r"\[\d{2}:\d{2}:\d{2}\]\s*\{'display_name':\s*'([^']*)'",
        transcript_text,
    ):
        name = m.group(1).strip()
        if name:
            participants.add(name)

    return {
        "title": title,
        "account_name": account_name,
        "date": date,
        "duration": duration,
        "participants": sorted(participants),
        "action_items": action_items,
        "transcript_text": transcript_text,
    }


def build_prompt(data, include_full_transcript=False):
    system_prompt = (
        "You are a JSON generator. Your job is to output a single JSON "
        "object. Do NOT output any text before or after the JSON. "
        "Do NOT think step by step. Do NOT explain your reasoning. "
        "Start directly with the opening brace '{'."
    )

    action_items_str = "\n".join(
        f"- {'[x]' if a['done'] else '[ ]'} {a['text']}"
        + (f" (assigned to: {a['assigned_to']})" if a["assigned_to"] else "")
        for a in data["action_items"]
    ) or "None"

    participants_str = ", ".join(data["participants"]) or "Unknown"

    if include_full_transcript:
        transcript_section = data["transcript_text"][:4000]
    else:
        t = data["transcript_text"]
        if len(t) > 1200:
            transcript_section = (
                t[:800]
                + "\n\n[... transcript truncated ...]\n\n"
                + t[-400:]
            )
        else:
            transcript_section = t

    user_prompt = f"""Call transcript:
Account: {data['account_name']}
Date: {data['date']}
Duration: {data['duration']}
Participants: {participants_str}
Title: {data['title']}

Action Items:
{action_items_str}

Transcript:
{transcript_section}

Respond ONLY with JSON: {{"key_insights":[],"risks":[],"opportunities":[],"sentiment":"","action_items":[],"recommended_next_steps":[],"health_score":5}}

health_score is 1-10 (1=critical risk, 10=perfect health).""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def call_lm_studio(messages, model="local-model", base_url="http://localhost:1234/v1"):
    client = OpenAI(base_url=base_url, api_key="not-needed", timeout=300.0)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        max_tokens=4000,
    )
    msg = response.choices[0].message
    text = (msg.content or "").strip()
    reasoning = (getattr(msg, "reasoning_content", None) or "").strip()
    if not text and reasoning:
        return reasoning
    return text


def call_claude(messages, model="claude-sonnet-4-20250514"):
    from anthropic import Anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "Error: ANTHROPIC_API_KEY not set. "
            "Create a .env file with ANTHROPIC_API_KEY=your_key"
        )
        sys.exit(1)

    client = Anthropic(api_key=api_key)

    system = None
    anthropic_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system = msg["content"]
        else:
            anthropic_messages.append({"role": msg["role"], "content": msg["content"]})

    response = client.messages.create(
        model=model,
        system=system,
        messages=anthropic_messages,
        max_tokens=2000,
        temperature=0.3,
    )
    return response.content[0].text


def parse_response(text):
    text = text.strip()

    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()

    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    return None


def process_file(path, args, output_dir):
    print(f"Processing: {path.name}...")

    try:
        data = parse_transcript(path)
    except Exception as e:
        print(f"  Error parsing {path.name}: {e}")
        return None

    messages = build_prompt(data, include_full_transcript=args.full_transcripts)

    try:
        if args.use_claude:
            response_text = call_claude(messages, model=args.claude_model)
        else:
            response_text = call_lm_studio(
                messages, model=args.model, base_url=args.lm_studio_url
            )
    except Exception as e:
        print(f"  LLM call failed for {path.name}: {e}")
        return None

    result = parse_response(response_text)
    if result is None:
        print(f"  Warning: Could not parse JSON for {path.name}. Saving raw.")
        result = {"_raw_response": response_text, "_parse_error": True}
    elif not isinstance(result, dict):
        print(f"  Warning: Unexpected response type for {path.name}. Saving raw.")
        result = {"_raw_response": response_text, "_parse_error": True}

    result["_file"] = path.name
    result["_account_name"] = data["account_name"]
    result["_call_date"] = data["date"]
    result["_participants"] = data["participants"]

    output_path = output_dir / f"{path.stem}.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"  -> {output_path}")

    return result


def build_aggregate(results, output_dir):
    print("\nBuilding aggregate report...")

    accounts = sorted(
        set(r.get("_account_name", "") for r in results if r)
    )
    health_scores = [
        r.get("health_score")
        for r in results
        if r and isinstance(r.get("health_score"), (int, float))
    ]

    overall = {
        "total_calls": len(results),
        "accounts": accounts,
        "average_health_score": (
            round(sum(health_scores) / len(health_scores), 1)
            if health_scores
            else None
        ),
    }

    aggregate = {"overall": overall, "calls": results}

    path = output_dir / "aggregate.json"
    path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(f"Aggregate report: {path}")


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Account Intelligence Tool — Analyze Fathom transcript files"
    )

    parser.add_argument(
        "--input-dir",
        default="./fathom_transcripts",
        help="Directory with Fathom transcript .md files (default: ./fathom_transcripts)",
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Directory for output JSON files (default: ./output)",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Build a single aggregate JSON from all results",
    )
    parser.add_argument(
        "--use-claude",
        action="store_true",
        help="Use Claude API instead of local LM Studio",
    )
    parser.add_argument(
        "--full-transcripts",
        action="store_true",
        help="Include full transcript text in prompt (default: condensed)",
    )
    parser.add_argument(
        "--model",
        default="local-model",
        help="Model name for LM Studio (default: local-model)",
    )
    parser.add_argument(
        "--claude-model",
        default="claude-sonnet-4-20250514",
        help="Claude model name (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--lm-studio-url",
        default="http://localhost:1234/v1",
        help="LM Studio base URL (default: http://localhost:1234/v1)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(
            f"Error: Input directory '{input_dir}' not found.\n"
            "Create it and add Fathom transcript .md files, "
            "or use --input-dir to point elsewhere."
        )
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.md"))
    if not files:
        print(f"No .md files found in '{input_dir}'.")
        sys.exit(0)

    print(f"Found {len(files)} transcript(s) in '{input_dir}'")
    print(f"Backend: {'Claude API' if args.use_claude else 'LM Studio (local)'}")
    print(f"Transcript: {'full' if args.full_transcripts else 'condensed'}")
    print()

    results = []
    for path in files:
        result = process_file(path, args, output_dir)
        if result is not None:
            results.append(result)

    if args.aggregate and results:
        build_aggregate(results, output_dir)

    print(
        f"\nDone. Processed {len(results)}/{len(files)} files. "
        f"Output in '{output_dir}/'."
    )


if __name__ == "__main__":
    main()
