# rebuild_aggregate.py
#
# Each run of transcript_analyst.py builds aggregate.json from only the
# files in that single batch. If you run the tool multiple times (different
# input dirs, retrying failures, etc.), each run overwrites the previous
# aggregate with partial data.
#
# This script scans ALL per-file JSONs in ./output/, rebuilds the full
# aggregate from every call across every batch, and writes it back.
# No LLM calls needed — just re-aggregation from existing results.

import json
from pathlib import Path

output_dir = Path("./output")
files = sorted(output_dir.glob("*.json"))
files = [f for f in files if f.name != "aggregate.json"]

results = []
for f in files:
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        results.append(data)
    except Exception as e:
        print(f"Skipped {f.name}: {e}")

accounts = sorted(set(r.get("_account_name", "") for r in results if r.get("_account_name")))
health_scores = [r.get("health_score") for r in results if isinstance(r.get("health_score"), (int, float))]

aggregate = {
    "overall": {
        "total_calls": len(results),
        "accounts": accounts,
        "average_health_score": round(sum(health_scores) / len(health_scores), 1) if health_scores else None
    },
    "calls": results
}

Path("./output/aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
print(f"Done. {len(results)} calls, {len(accounts)} accounts, avg health {aggregate['overall']['average_health_score']}")
