"""
Post-process decadal downscaling inference NetCDF outputs.

Direct analogue of scripts/analyze_inference_outputs.py.
Input variable names and output NetCDF structure are identical so all
visualization functions are shared verbatim.

Input files:  inference_run_*_lead_*.nc
              (written by infer_dec_downscale.py)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── reuse the original analysis script instead of duplicating code ────────────
# The output format from infer_dec_downscale.py is intentionally identical to
# scripts/infer.py so we can delegate to the same analysis functions.

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analyze_inference_outputs import (   # noqa: E402  (deferred import)
    collate_monthly,
    save_collated,
    create_six_panel_maps,
    create_reliability_plots,
    create_scatter_plots,
    create_taylor_diagrams,
    create_qq_plots,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate and visualize decadal downscaling inference outputs"
    )
    parser.add_argument("--input-dir",  required=True,
                        help="Directory containing inference_run_*_lead_*.nc files")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: <input-dir>/analysis)")
    parser.add_argument("--pattern", default="inference_run_*_lead_*.nc")
    parser.add_argument("--reliability-bins", type=int, default=20)
    parser.add_argument("--dpi",   type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir  = Path(args.input_dir).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir = (
        Path(args.output_dir).resolve() if args.output_dir
        else (input_dir / "analysis")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{args.pattern}' in {input_dir}"
        )

    print(f"Found {len(files)} inference files")

    collated = collate_monthly(files)
    collated_path = output_dir / "monthly_collated.nc"
    save_collated(collated, collated_path)
    print(f"Saved collated monthly NetCDF: {collated_path}")

    create_six_panel_maps(collated, output_dir=output_dir, dpi=args.dpi)
    print(f"Saved 6-panel maps:       {output_dir / 'maps_6panel'}")

    create_reliability_plots(
        collated, output_dir=output_dir, nbins=args.reliability_bins, dpi=args.dpi
    )
    print(f"Saved reliability plots:  {output_dir / 'reliability'}")

    create_scatter_plots(collated, output_dir=output_dir, dpi=args.dpi)
    print(f"Saved scatter plots:      {output_dir / 'scatter'}")

    create_taylor_diagrams(collated, output_dir=output_dir, dpi=args.dpi)
    print(f"Saved Taylor diagrams:    {output_dir / 'taylor'}")

    create_qq_plots(collated, output_dir=output_dir, dpi=args.dpi)
    print(f"Saved Q-Q plots:          {output_dir / 'qq_plots'}")


if __name__ == "__main__":
    main()
