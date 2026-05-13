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
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

_ENC = getattr(sys.stdout, "encoding", "")
_SUPPORTS_UNICODE = _ENC and _ENC.lower() in ("utf-8", "utf8", "utf-16", "unicode")

SEP = "\u2500" if _SUPPORTS_UNICODE else "-"
CHAR_X = "\u2717" if _SUPPORTS_UNICODE else "x"
CHAR_WARN = "\u26a0" if _SUPPORTS_UNICODE else "!"
CHAR_ARROW = "\u2192" if _SUPPORTS_UNICODE else "->"


def format_elapsed(seconds):
    if seconds < 60:
        return f"{int(seconds)}s"
    return f"{int(seconds // 60)}m {int(seconds % 60)}s"


def print_inline(text):
    sys.stdout.write(text)
    sys.stdout.flush()


def map_llm_error(e, args):
    msg = str(e).lower()
    if args.use_claude:
        if "api_key" in msg or "auth" in msg or "unauthorized" in msg:
            return "Claude API authentication failed"
        if "rate" in msg and "limit" in msg:
            return "Claude API rate limit exceeded"
        return f"Claude API error: {e}"
    else:
        url = args.lm_studio_url.replace("http://", "").replace("/v1", "")
        if "connection refused" in msg or "connection error" in msg or "cannot connect" in msg:
            return f"LM Studio not responding ({url})"
        if "timeout" in msg or "timed out" in msg:
            return f"LM Studio timed out ({args.timeout}s)"
        if "model not found" in msg or ("model" in msg and "not found" in msg):
            return f"Model '{args.model}' not found in LM Studio"
        return f"LM Studio error: {e}"


def map_llm_tip(e, args):
    msg = str(e).lower()
    if args.use_claude:
        if "api_key" in msg or "auth" in msg:
            return "check your ANTHROPIC_API_KEY in .env"
        if "rate" in msg and "limit" in msg:
            return "wait a moment and try again"
        return None
    else:
        if "connection refused" in msg or "connection error" in msg:
            return "check LM Studio is running and a model is loaded"
        if "timeout" in msg or "timed out" in msg:
            return "the model may be too large for your hardware; try a smaller model"
        if "model not found" in msg:
            return "check the model name with --model (default: local-model)"
        return None


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


def call_lm_studio(messages, model="local-model", base_url="http://localhost:1234/v1", timeout=300.0):
    client = OpenAI(base_url=base_url, api_key="not-needed", timeout=timeout)
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


_SELF_NAMES = {"marck", "marck ernest", "passion marck", "marck passion"}
_INTERNAL_NAMES: set = {}
_NAME_FILTERS: set = {"iphone", "ipad", "rta"}


def _load_config_lines(filename):
    path = Path(filename)
    if not path.exists():
        return []
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            lines.append(line.lower())
    return lines


def _strip_filters(name, filters):
    """Remove employer filter words and suffixes from name while preserving case."""
    result = name.strip()
    result = result.replace("\u2018", "'").replace("\u2019", "'")
    result = result.replace("\u201c", '"').replace("\u201d", '"')
    for word in sorted(filters, key=len, reverse=True):
        result = re.sub(r"\b" + re.escape(word) + r"\b", "", result, flags=re.IGNORECASE).strip()
    result = re.sub(r"\s*\|.*", "", result).strip()
    result = re.sub(r"\s*\([^)]*\)\s*$", "", result).strip()
    result = re.sub(r"\bU\+\s*", "", result, flags=re.IGNORECASE).strip()
    result = re.sub("\\s+[-\u2014\u2022]\\s+\\S+\\.\\S+\\s*$", "", result).strip()
    result = re.sub("^\\S+\\.\\S+\\s+[-\u2014\u2022]\\s+", "", result).strip()
    result = re.sub(r"'s$", "", result).strip()
    result = result.rstrip(".,").strip()
    result = re.sub(r"[,@\s]+", " ", result).strip()
    return result


def _clean_display_name(name, filters):
    """Lowercased version for matching."""
    return _strip_filters(name, filters).lower()


def _strip_special(text):
    """Remove trademark/copyright-like symbols for name matching."""
    return re.sub(r"[\\u00ae\\u2122\\u00a9\u00ae\u2122\u00a9]", "", text)


_CANONICAL_NAMES = {
    "rebecca welsh": "Rebecca Mitchell Welsh",
    "rebecca mitchell welsh": "Rebecca Mitchell Welsh",
    "stan c": "Stanley Campbell",
    "stan campbell": "Stanley Campbell",
    "stanley campbell": "Stanley Campbell",
    "stace saul": "Stacey Saul",
    "stacey fc": "Stacey Saul",
    "stacey marie saul": "Stacey Saul",
    "stacey saul": "Stacey Saul",
    "amy terry": "Amy Meyer-Terry",
    "moera": "Mo\u00ebra Saule",
    "kristin mcdermott": "Kristin MacDermott",
    "michael mcdermott": "Michael MacDermott",
    "macylemos": "Macy Lemos",
    "equi-tape\u00ae equine kinesiology tape and education": "EquiTecs",
    "equitecs - equine technologies institute": "EquiTecs",
}


def _clean_title_hint(title):
    """Strip filter words, self-names and noise from a title to find name tokens."""
    if not title:
        return ""
    cleaned = title.lower().strip()
    for word in sorted(_NAME_FILTERS, key=len, reverse=True):
        cleaned = re.sub(r"\b" + re.escape(word) + r"\b", "", cleaned, flags=re.IGNORECASE).strip()
    for name in _SELF_NAMES:
        cleaned = re.sub(r"\b" + re.escape(name) + r"\b", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"[^\w\s'-]", " ", cleaned)
    cleaned = re.sub(r"\b(and|with|w|reschedule|kickoff|call|review|sesh|post|launch|prelaunch|dev|apple|account|final|follow|up|check|in|work|stuff)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bw/?\S*", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _find_primary_from_title(title, participants):
    """Return the first participant whose name matches the cleaned title hint."""
    cleaned = _clean_title_hint(title)
    if not cleaned:
        return None
    cleaned_words = set(w for w in cleaned.split() if len(w) > 1)

    for p in participants:
        p_lower = _strip_special(p.strip().lower())
        first_name = p_lower.split()[0] if p_lower.split() else ""
        if not first_name or len(first_name) <= 1:
            continue
        if first_name in cleaned_words:
            return p

    for p in participants:
        p_lower = _strip_special(p.strip().lower())
        first_name = p_lower.split()[0] if p_lower.split() else ""
        if not first_name or len(first_name) <= 2:
            continue
        for cw in cleaned_words:
            norm_cw = _strip_special(cw)
            if norm_cw.startswith(first_name) or first_name.startswith(norm_cw):
                return p

    return None


def derive_account_name(participants, title=""):
    # Filter out self and internal names
    filtered = []
    for p in participants:
        lower = p.strip().lower()
        if lower in _SELF_NAMES or lower in _INTERNAL_NAMES:
            continue
        cleaned = _clean_display_name(p, _NAME_FILTERS)
        if not cleaned or len(cleaned) <= 2:
            continue
        if any(re.search(r"\b" + re.escape(n) + r"\b", cleaned) for n in _INTERNAL_NAMES):
            continue
        filtered.append(p)

    # Title-based primary detection
    if title and filtered:
        primary = _find_primary_from_title(title, filtered)
        if primary:
            display = _strip_filters(primary, _NAME_FILTERS)
            if display.islower():
                display = display[0].upper() + display[1:]
            display_lower = display.lower()
            if display_lower in _CANONICAL_NAMES:
                display = _CANONICAL_NAMES[display_lower]
            return display if display else "(solo)"

    # Fallback: joint account name from all non-excluded participants
    seen = set()
    result = []
    for p in filtered:
        display = _strip_filters(p, _NAME_FILTERS)
        if display.islower():
            display = display[0].upper() + display[1:]
        display_lower = display.lower()
        if display_lower in _CANONICAL_NAMES:
            display = _CANONICAL_NAMES[display_lower]
        if display.lower() not in seen:
            seen.add(display.lower())
            result.append(display)
    return ", ".join(result) if result else "(solo)"


def process_file(path, args, output_dir, index, total):
    print(f"[{index}/{total}] {path.name}")

    try:
        data = parse_transcript(path)
    except Exception as e:
        print(f"      {CHAR_X} Parse error: {e}")
        return None

    messages = build_prompt(data, include_full_transcript=args.full_transcripts)
    start = time.time()
    print_inline(f"      Analysing...")

    stop_timer = threading.Event()

    def _timer():
        while not stop_timer.is_set():
            elapsed = int(time.time() - start)
            print_inline(f"\r      Analysing... ({format_elapsed(elapsed)})")
            stop_timer.wait(1)

    timer_thread = threading.Thread(target=_timer, daemon=True)
    timer_thread.start()

    try:
        if args.use_claude:
            response_text = call_claude(messages, model=args.claude_model)
        else:
            response_text = call_lm_studio(
                messages, model=args.model, base_url=args.lm_studio_url,
                timeout=args.timeout,
            )
    except Exception as e:
        stop_timer.set()
        timer_thread.join(1)
        elapsed = int(time.time() - start)
        print(f"\r      Analysing... {CHAR_X} {map_llm_error(e, args)}")
        tip = map_llm_tip(e, args)
        if tip:
            print(f"      Tip: {tip}")
        return None

    stop_timer.set()
    timer_thread.join(1)
    elapsed = int(time.time() - start)

    result = parse_response(response_text)
    health_str = ""
    if result is None:
        print(f"\r      Analysing... done ({format_elapsed(elapsed)})  {CHAR_WARN} Could not parse JSON")
        result = {"_raw_response": response_text, "_parse_error": True}
    elif not isinstance(result, dict):
        print(f"\r      Analysing... done ({format_elapsed(elapsed)})  {CHAR_WARN} Unexpected response type")
        result = {"_raw_response": response_text, "_parse_error": True}
    else:
        print(f"\r      Analysing... done ({format_elapsed(elapsed)})")
        health = result.get("health_score")
        if isinstance(health, (int, float)):
            health_str = f"health: {health}/10"
            if health < 5:
                health_str += f"  {CHAR_WARN} low"

    result["_file"] = path.name
    result["_account_name"] = derive_account_name(data["participants"], title=data.get("title", ""))
    result["_call_date"] = data["date"]
    result["_participants"] = data["participants"]

    output_path = output_dir / f"{path.stem}.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"      {CHAR_ARROW} {output_path}  {health_str}")

    return result


def build_aggregate(results, output_dir):
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


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Account Intelligence Tool — Analyze call transcript files"
    )

    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Process a single transcript file (overrides --input-dir)",
    )
    parser.add_argument(
        "--input-dir",
        default="./raw_transcripts",
        help="Directory with transcript .md files (default: ./raw_transcripts)",
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Directory for output JSON files (default: ./output)",
    )
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Skip building aggregate.json (default: aggregate is generated)",
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
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds for LM Studio API calls (default: 300)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process transcripts even if output already exists",
    )
    parser.add_argument(
        "--self-name",
        action="append",
        default=[],
        help="Your name as it appears in _participants (repeatable, e.g. --self-name 'Marck' --self-name 'Marck Ernest')",
    )
    parser.add_argument(
        "--exclude-name",
        action="append",
        default=[],
        help="Internal team member to exclude from _account_name (repeatable)",
    )
    parser.add_argument(
        "--filter-word",
        action="append",
        default=[],
        help="Word/phrase to strip from participant display names (repeatable)",
    )

    args = parser.parse_args()
    _SELF_NAMES.update(_load_config_lines(".self_names"))
    _SELF_NAMES.update(n.strip().lower() for n in args.self_name)
    _INTERNAL_NAMES.update(_load_config_lines(".exclude_names"))
    _INTERNAL_NAMES.update(n.strip().lower() for n in args.exclude_name)
    _NAME_FILTERS.update(_load_config_lines(".filter_words"))
    _NAME_FILTERS.update(n.strip().lower() for n in args.filter_word)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(
            f"Error: Input directory '{input_dir}' not found.\n"
            "Create it and add transcript .md files, "
            "or use --input-dir to point elsewhere."
        )
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File '{file_path}' not found.")
            sys.exit(1)
        all_files = [file_path]
        input_dir = file_path.parent
    else:
        all_files = sorted(input_dir.glob("*.md"))
        if not all_files:
            print(f"No .md files found in '{input_dir}'.")
            sys.exit(0)

    new_files = []
    skipped = 0
    for path in all_files:
        output_path = output_dir / f"{path.stem}.json"
        if output_path.exists() and not args.force:
            skipped += 1
        else:
            new_files.append(path)

    total = len(all_files)
    new_count = len(new_files)

    url = args.lm_studio_url.replace("http://", "").replace("/v1", "")
    print("Account Intelligence Tool")
    print(SEP * 60)
    if skipped:
        print(f"Transcripts: {total} found in '{input_dir}'")
        print(f"             {new_count} new, {skipped} already processed (--force to re-process)")
    else:
        print(f"Transcripts: {total} found in '{input_dir}'")
    print(f"Backend:     {'Claude API' if args.use_claude else f'LM Studio (local) - {url}'}")
    print(f"Mode:        {'full' if args.full_transcripts else 'condensed'}")

    if new_count == 0:
        print()
        print(SEP * 60)
        print(f"Done  0/{total} processed - all files already in '{output_dir}/'")
        print(f"      use --force to re-process")
        sys.exit(0)

    print()

    results = []
    failures = 0
    for i, path in enumerate(new_files, 1):
        result = process_file(path, args, output_dir, i, new_count)
        if result is not None:
            results.append(result)
        else:
            failures += 1

    if not args.no_aggregate and results:
        build_aggregate(results, output_dir)

    health_scores = [
        r.get("health_score")
        for r in results
        if r and isinstance(r.get("health_score"), (int, float))
    ]
    avg_health = round(sum(health_scores) / len(health_scores), 1) if health_scores else None

    print()
    print(SEP * 60)
    summary = f"Done  {len(results)}/{new_count} processed"
    if avg_health is not None:
        summary += f" - avg health {avg_health}"
    summary += f" - output in '{output_dir}/'"
    print(summary)
    if failures:
        print(f"      {failures} failed - re-run with --file <path> to retry individual files")
    if skipped:
        print(f"      {skipped} skipped (already processed)")


if __name__ == "__main__":
    main()
