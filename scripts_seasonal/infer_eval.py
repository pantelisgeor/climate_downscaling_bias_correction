#!/usr/bin/env python3
"""
Seasonal ClimateNet – combined inference + evaluation script.

Seasonal analogue of  scripts/infer_eval.sh  (decadal), written as a Python
script for portability and richer error handling.

What it does
------------
1.  Runs  scripts_seasonal/infer_seasonal.py
    → saves one NetCDF per (lead_month, year) with all 25 members as a
      dimension, under  <output-dir>/  as  inference_lead_{L}_year_{Y}.nc
2.  Runs  scripts_seasonal/analyze_seasonal_outputs.py
    → reads those NetCDFs, computes monthly aggregates, and writes maps +
      scatter / bias / Taylor diagrams under  <analysis-dir>/

Both steps are run in-process (via subprocess) so that each retains its own
argument namespace.  The script exits immediately if either step fails.

Usage examples
--------------
# Simplest – point at an experiment directory, infer test years:
    python scripts_seasonal/infer_eval.py \\
        --experiment-dir /path/to/exp_s01_cnn_tweedie_seasonal

# Override years, output dirs, device:
    python scripts_seasonal/infer_eval.py \\
        --experiment-dir /path/to/exp_s01_cnn_tweedie_seasonal \\
        --checkpoint best \\
        --years 2017,2018,2019,2020 \\
        --infer-dir /scratch/inferred \\
        --analysis-dir /scratch/analysis \\
        --device cuda:0 \\
        --batch-size 512 \\
        --dpi 300

# Inference only (skip analysis):
    python scripts_seasonal/infer_eval.py \\
        --experiment-dir /path/to/exp_s01 \\
        --skip-analysis

# Analysis only (skip inference, reuse existing NetCDFs):
    python scripts_seasonal/infer_eval.py \\
        --experiment-dir /path/to/exp_s01 \\
        --skip-infer
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────────────

def _banner(msg: str) -> None:
    print()
    print("=" * 80)
    print(msg)
    print("=" * 80)


def _run(cmd: list[str], label: str) -> None:
    """Run *cmd* as a subprocess; raise on non-zero exit."""
    _banner(label)
    print("Command:", " ".join(str(c) for c in cmd))
    print()
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\nERROR: '{label}' failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)
    print(f"\n{label} completed in {elapsed:.1f}s")


def _resolve_test_years(experiment_dir: Path) -> str | None:
    """
    Try to read test_years from the experiment's saved config.yaml.
    Returns a comma-separated string like '2017,2018,2019,2020', or None.
    """
    config_path = experiment_dir / "config.yaml"
    if not config_path.exists():
        return None
    try:
        import yaml  # only needed here
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        years = cfg.get("data", {}).get("test_years", [])
        if years:
            return ",".join(str(y) for y in years)
    except Exception:
        pass
    return None


# ── argument parser ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seasonal ClimateNet: inference + evaluation in one step.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── required ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--experiment-dir", required=True,
        help="Experiment directory containing config.yaml and checkpoints/.",
    )

    # ── inference options ─────────────────────────────────────────────────────
    parser.add_argument(
        "--checkpoint", default="best",
        help="Checkpoint spec: 'best', 'last', or explicit checkpoint path.",
    )
    parser.add_argument(
        "--years", default=None,
        help=(
            "Years to run inference on, e.g. '2017,2018,2020-2022'.  "
            "Defaults to data.test_years from the experiment config.yaml."
        ),
    )
    parser.add_argument(
        "--infer-dir", default=None,
        help=(
            "Directory to write inference NetCDF files.  "
            "Defaults to <experiment-dir>/inferred."
        ),
    )
    parser.add_argument(
        "--device", default=None,
        help="Device override, e.g. 'cpu', 'cuda', 'cuda:0'.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Batch size for the model forward pass.",
    )

    # ── analysis options ──────────────────────────────────────────────────────
    parser.add_argument(
        "--analysis-dir", default=None,
        help=(
            "Directory to write analysis outputs.  "
            "Defaults to <experiment-dir>/analysis."
        ),
    )
    parser.add_argument(
        "--pattern", default="inference_lead_*_year_*.nc",
        help="Glob pattern for inference NetCDF files passed to the analysis script.",
    )
    parser.add_argument(
        "--reliability-bins", type=int, default=20,
        help="Number of bins for reliability/bias plots.",
    )
    parser.add_argument(
        "--dpi", type=int, default=200,
        help="Figure DPI for saved plots.",
    )

    # ── step toggles ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--skip-infer", action="store_true",
        help="Skip the inference step (useful when NetCDF files already exist).",
    )
    parser.add_argument(
        "--skip-analysis", action="store_true",
        help="Skip the analysis step (inference only).",
    )

    return parser.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── resolve paths ─────────────────────────────────────────────────────────
    experiment_dir = Path(args.experiment_dir).resolve()
    if not experiment_dir.exists():
        print(f"ERROR: experiment directory not found: {experiment_dir}", file=sys.stderr)
        sys.exit(1)

    infer_dir    = Path(args.infer_dir).resolve()    if args.infer_dir    else experiment_dir / "inferred"
    analysis_dir = Path(args.analysis_dir).resolve() if args.analysis_dir else experiment_dir / "analysis"

    # ── resolve years ─────────────────────────────────────────────────────────
    years = args.years
    if years is None and not args.skip_infer:
        years = _resolve_test_years(experiment_dir)
        if years is None:
            print(
                "ERROR: --years not provided and could not be read from config.yaml.\n"
                "       Please supply --years, e.g. --years 2017,2018,2019,2020.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Auto-detected test years from config: {years}")

    # ── resolve this script's directory to find sibling scripts ───────────────
    scripts_dir = Path(__file__).parent.resolve()
    infer_script    = scripts_dir / "infer_seasonal.py"
    analysis_script = scripts_dir / "analyze_seasonal_outputs.py"

    for p in [infer_script, analysis_script]:
        if not p.exists():
            print(f"ERROR: required sibling script not found: {p}", file=sys.stderr)
            sys.exit(1)

    python = sys.executable   # same interpreter that's running this script

    # ── summary ───────────────────────────────────────────────────────────────
    _banner("SEASONAL CLIMATENET  –  INFER + EVAL")
    print(f"  Experiment : {experiment_dir}")
    print(f"  Checkpoint : {args.checkpoint}")
    if not args.skip_infer:
        print(f"  Years      : {years}")
        print(f"  Infer dir  : {infer_dir}")
    if not args.skip_analysis:
        print(f"  Analysis   : {analysis_dir}")
    print(f"  Skip infer : {args.skip_infer}")
    print(f"  Skip anal. : {args.skip_analysis}")

    total_t0 = time.time()

    # ── step 1: inference ─────────────────────────────────────────────────────
    if not args.skip_infer:
        infer_cmd = [
            python, str(infer_script),
            "--experiment-dir", str(experiment_dir),
            "--checkpoint",     args.checkpoint,
            "--years",          years,
            "--output-dir",     str(infer_dir),
            "--batch-size",     str(args.batch_size),
        ]
        if args.device:
            infer_cmd += ["--device", args.device]

        _run(infer_cmd, "STEP 1 / 2 — Inference (infer_seasonal.py)")
    else:
        _banner("STEP 1 / 2 — Inference  [SKIPPED]")

    # ── step 2: analysis ──────────────────────────────────────────────────────
    if not args.skip_analysis:
        analysis_cmd = [
            python, str(analysis_script),
            "--input-dir",        str(infer_dir),
            "--output-dir",       str(analysis_dir),
            "--pattern",          args.pattern,
            "--reliability-bins", str(args.reliability_bins),
            "--dpi",              str(args.dpi),
        ]

        _run(analysis_cmd, "STEP 2 / 2 — Analysis (analyze_seasonal_outputs.py)")
    else:
        _banner("STEP 2 / 2 — Analysis  [SKIPPED]")

    # ── done ──────────────────────────────────────────────────────────────────
    elapsed = time.time() - total_t0
    _banner(f"ALL DONE  ({elapsed:.1f}s total)")
    print(f"  Inference NetCDFs : {infer_dir}")
    if not args.skip_analysis:
        print(f"  Analysis outputs  : {analysis_dir}")
    print()


if __name__ == "__main__":
    main()
