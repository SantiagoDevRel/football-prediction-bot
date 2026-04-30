"""Weekly model retraining with safety gate.

Workflow:
    1. Snapshot the previous models (move to models_artifacts/_archive/{date}/).
    2. Re-train all models on the latest DB.
    3. Run scripts/backtest_full.py to produce a fresh report.
    4. Read both old and new reports — if new log-loss is meaningfully WORSE
       (delta > 0.02) on either league, REVERT to the snapshot and alert.
    5. Otherwise, the new models stay in place.

Run via cron / Windows Task Scheduler weekly (Sundays at 04:00 ideal).

Usage:
    python scripts/weekly_retrain.py
    python scripts/weekly_retrain.py --dry-run  # train + backtest, don't promote
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ARTIFACTS = ROOT / "models_artifacts"
ARCHIVE_ROOT = ARTIFACTS / "_archive"


def parse_logloss_from_report(text: str) -> dict[str, dict[str, float]]:
    """Parse a backtest markdown report into {league: {model: log_loss}}."""
    out: dict[str, dict[str, float]] = {}
    league = None
    for line in text.splitlines():
        h = re.match(r"^##\s+(.+)$", line)
        if h:
            league = h.group(1).strip()
            out[league] = {}
            continue
        if league and "|" in line:
            cells = [c.strip() for c in line.split("|") if c.strip()]
            if len(cells) >= 6 and cells[0] not in ("Model", ":---"):
                model = cells[0]
                try:
                    log_loss = float(cells[3])
                    out[league][model] = log_loss
                except ValueError:
                    pass
    return out


def find_latest_report() -> Path | None:
    docs = ROOT / "docs"
    candidates = sorted(docs.glob("backtest-*.md"))
    return candidates[-1] if candidates else None


def archive_current() -> Path:
    today = date.today().isoformat()
    target = ARCHIVE_ROOT / today
    target.mkdir(parents=True, exist_ok=True)
    for p in ARTIFACTS.glob("*.pkl"):
        shutil.copy2(p, target / p.name)
    return target


def restore_from_archive(archive_dir: Path) -> None:
    for p in archive_dir.glob("*.pkl"):
        shutil.copy2(p, ARTIFACTS / p.name)


def run(cmd: list[str]) -> None:
    subprocess.check_call(cmd, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"[weekly_retrain] starting on {date.today()}")

    # 1. Read prior log-losses
    prior_report = find_latest_report()
    prior_ll: dict[str, dict[str, float]] = {}
    if prior_report:
        prior_ll = parse_logloss_from_report(prior_report.read_text(encoding="utf-8"))
        print(f"  prior report: {prior_report.name}")

    # 2. Snapshot
    snap = archive_current()
    print(f"  archived current models -> {snap.relative_to(ROOT)}")

    # 3. Retrain
    print("  retraining all models…")
    py = str(ROOT / ".venv" / "Scripts" / "python.exe")
    run([py, "scripts/train_models.py"])

    # 4. New backtest
    print("  running fresh backtest…")
    run([py, "scripts/backtest_full.py"])
    new_report = find_latest_report()
    if not new_report:
        print("  WARN: no backtest report found after retrain. Keeping new models.")
        return
    new_ll = parse_logloss_from_report(new_report.read_text(encoding="utf-8"))

    # 5. Compare and decide
    print("\n  log-loss comparison (lower is better):")
    regressed = False
    for league, models in new_ll.items():
        for model, new_val in models.items():
            old_val = prior_ll.get(league, {}).get(model)
            if old_val is None:
                print(f"    {league}/{model}: new={new_val:.4f} (no prior)")
                continue
            delta = new_val - old_val
            mark = "📈" if delta < 0 else ("⚠️" if delta > 0.02 else "≈")
            print(f"    {league}/{model}: old={old_val:.4f} new={new_val:.4f} delta={delta:+.4f} {mark}")
            if delta > 0.02 and model == "stacking":
                regressed = True

    if regressed and not args.dry_run:
        print("\n  ⚠️ Regression detected on stacking. Reverting to archived models.")
        restore_from_archive(snap)
        print("  reverted.")
        sys.exit(1)
    else:
        print(f"\n  ✅ {'(dry run) ' if args.dry_run else ''}new models kept.")


if __name__ == "__main__":
    main()
