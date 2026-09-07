# sensitivity/run_all.py
"""Runs all 12 sensitivity configs and writes sensitivity/outputs/summary.json."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sensitivity.configs import RUN_CONFIGS
from sensitivity.lib.dataset_builder import build_dataset_for_config, build_fundamentals_table
from sensitivity.lib.train_evaluate import train_and_evaluate

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fundamentals = build_fundamentals_table()

    summary = []
    for config in RUN_CONFIGS:
        run_dir = OUTPUT_DIR / config["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)

        dataset = build_dataset_for_config(config, fundamentals)
        dataset.to_csv(run_dir / "dataset.csv", index=False)

        result = train_and_evaluate(dataset, config)
        with (run_dir / "metrics.json").open("w") as f:
            json.dump(result, f, indent=2, default=float)

        summary.append({
            "run_id": config["run_id"],
            "description": config["description"],
            "dimension": config["dimension"],
            "value_label": config["value_label"],
            "row_count": result["row_count"],
            "best_model": result["best_model"],
            "best_accuracy": result["best_accuracy"],
            "baseline_accuracy": result["baseline_accuracy"],
            "beats_baseline": result.get("beats_baseline"),
        })
        print(f"{config['run_id']}: rows={result['row_count']} "
              f"best={result['best_model']} acc={result['best_accuracy']} "
              f"baseline={result['baseline_accuracy']}")

    with (OUTPUT_DIR / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nWrote summary for {len(summary)} runs to {OUTPUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
