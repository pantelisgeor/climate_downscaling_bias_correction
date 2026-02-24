#!/usr/bin/env python3
"""
Preprocess the seasonal NetCDF into the canonical structure expected by
SeasonalDataLoader.

Two structural fixes:

  1. tp dimension order
       Raw       : ('number', 'latitude', 'longitude', 'time')
       Canonical : ('time',   'number',   'latitude',  'longitude')

  2. lead_time variable
       Raw       : timedelta64[ns]  (time, number, lat, lon)   ~25 GB
       Canonical : int8             (time, number)              ~2 MB
       The spatial dimensions are spatially uniform, so a single point
       [:, :, 0, 0] is sufficient — no spatial averaging needed.

Why this script uses netCDF4 directly (not xarray/dask)
--------------------------------------------------------
tp is stored as (number, lat, lon, time) with on-disk chunks (4, 4, 9, 14167).
Time is the LAST dimension.  To write output chunks aligned to time
(1000, 25, 28, 53), every dask chunk read would need to decompress the
entire file — making xarray/dask ~50x slower than reading the whole array
into RAM once and transposing it.

  tp   :  25 x 28 x 53 x 85000 x 4 B  ~12.6 GB  (fits in available RAM)
  lead :       85000 x 25      x 8 B  ~   17 MB  (read single spatial point)

All other variables already have time as the first dimension and are copied
in streaming 1000-timestep slices — no RAM spike.

Usage
-----
    # Inspect only (no writes)
    python scripts_seasonal/preprocess_seasonal_nc.py \\
        --nc_path /nvme/h/pgeorgiades/scratch/training_seasonal.nc --dry-run

    # Fix in-place  (~5-10 min)
    python scripts_seasonal/preprocess_seasonal_nc.py \\
        --nc_path /nvme/h/pgeorgiades/scratch/training_seasonal.nc

The script is idempotent: if the file is already canonical it exits without
touching it.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

_DAYS_PER_MONTH = 30.4375  # 365.25 / 12

_FC_VARS   = ["t2m", "hurs", "tmax", "tp"]
_TIME_VARS = ["tasERA", "tasmaxERA", "tpERA", "rhERA",
              "dem", "rho", "phi", "sin_time", "cos_time"]
_CCI_VAR   = "cci_agg"

_T_CHUNK = 1000   # output time steps per chunk


def inspect(nc_path: Path) -> dict:
    """Detect which fixes are needed.  Never modifies the file."""
    import netCDF4 as nc4

    print(f"\n{'='*70}")
    print(f"INSPECTING: {nc_path}")
    print(f"  File size: {nc_path.stat().st_size / 1e9:.2f} GB")
    print(f"{'='*70}")

    ds = nc4.Dataset(nc_path, "r")

    # ── check 1: tp dimension order ──────────────────────────────────────────
    canonical_tp = ("time", "number", "latitude", "longitude")
    if "tp" in ds.variables:
        actual_tp = ds.variables["tp"].dimensions
        tp_bad = actual_tp != canonical_tp
        print(f"\n  tp dims:      {actual_tp}")
        print(f"  tp canonical: {canonical_tp}")
        print(f"  tp OK:        {not tp_bad}")
    else:
        tp_bad = False
        print(f"\n  tp: NOT FOUND — skipping.")

    # ── check 2: lead_time → lead_month ──────────────────────────────────────
    has_lead_time  = "lead_time"  in ds.variables
    has_lead_month = "lead_month" in ds.variables
    lead_bad = has_lead_time and not has_lead_month
    print(f"\n  lead_time  present: {has_lead_time}")
    print(f"  lead_month present: {has_lead_month}")
    if has_lead_month:
        arr = ds.variables["lead_month"][:]
        print(f"  lead_month dtype: {arr.dtype}  "
              f"dims={ds.variables['lead_month'].dimensions}  "
              f"range [{int(arr.min())}–{int(arr.max())}]")
    print(f"  lead OK: {not lead_bad}")

    needs_work = tp_bad or lead_bad
    print(f"\n{'─'*70}")
    if not needs_work:
        print("  ✓  File is already in canonical form — no changes needed.")
    else:
        if tp_bad:
            print("  ✗  tp needs transposing")
        if lead_bad:
            print("  ✗  lead_time needs converting to lead_month (int8, time×number)")
    print(f"{'─'*70}\n")

    ds.close()
    return {"tp_needs_transpose": tp_bad, "lead_needs_convert": lead_bad,
            "needs_work": needs_work}


def preprocess(nc_path: Path) -> None:
    """
    Rewrite nc_path in canonical form using netCDF4 directly.
    Writes to a .tmp sibling; renames atomically only on success.
    """
    import netCDF4 as nc4

    tmp_path = nc_path.with_suffix(".nc.tmp")
    print(f"\n{'='*70}")
    print(f"PREPROCESSING")
    print(f"  Input : {nc_path}  ({nc_path.stat().st_size/1e9:.2f} GB)")
    print(f"  Output: {tmp_path}")
    print(f"{'='*70}\n")

    src = nc4.Dataset(nc_path, "r")
    dst = nc4.Dataset(tmp_path, "w", format="NETCDF4")

    n_time   = len(src.dimensions["time"])
    n_number = len(src.dimensions["number"])
    n_lat    = len(src.dimensions["latitude"])
    n_lon    = len(src.dimensions["longitude"])
    n_cci    = len(src.dimensions["cci_class"])

    # ── 1. copy dimensions ─────────────────────────────────────────────────
    print("Step 1/5 — dimensions …")
    for dname, dim in src.dimensions.items():
        dst.createDimension(dname, len(dim))
    print(f"  {list(dst.dimensions.keys())}")

    # ── 2. copy coordinate variables ────────────────────────────────────────
    print("\nStep 2/5 — coordinate variables …")
    for vname in ["time", "number", "latitude", "longitude", "cci_class"]:
        if vname not in src.variables:
            continue
        sv = src.variables[vname]
        dv = dst.createVariable(vname, sv.dtype, sv.dimensions, zlib=False)
        dv.setncatts({k: sv.getncattr(k) for k in sv.ncattrs()})
        dv[:] = sv[:]
        print(f"  {vname}")

    # ── 3. lead_time → lead_month ────────────────────────────────────────────
    print("\nStep 3/5 — converting lead_time → lead_month …")
    t0 = time.time()
    if "lead_time" in src.variables and "lead_month" not in src.variables:
        lt_var = src.variables["lead_time"]
        # The field is spatially uniform — read a single point to get (time, number).
        # [:, :, 0, 0] = only 85000 × 25 × 8 B = 17 MB instead of 25 GB.
        if "latitude" in lt_var.dimensions:
            lt_slice = lt_var[:, :, 0, 0]          # (time, number)
        else:
            lt_slice = lt_var[:, :]                 # already (time, number)
        td_ns = np.array(lt_slice, dtype=np.int64)  # timedelta64[ns] as int64
        months_arr = np.round(
            td_ns / 1e9 / 86400.0 / _DAYS_PER_MONTH
        ).astype(np.int8)
        print(f"  range: {int(months_arr.min())}–{int(months_arr.max())}  "
              f"({time.time()-t0:.1f}s, read {lt_slice.nbytes/1e6:.0f} MB)")
        lm = dst.createVariable(
            "lead_month", "i1", ("time", "number"),
            zlib=False, chunksizes=(n_time, n_number),
        )
        lm.long_name = "Forecast lead time in months (0-6)"
        lm.units = "months"
        lm[:] = months_arr
        print(f"  lead_month written.")
    elif "lead_month" in src.variables:
        sv = src.variables["lead_month"]
        dv = dst.createVariable("lead_month", "i1", ("time", "number"),
                                zlib=False, chunksizes=(n_time, n_number))
        dv.setncatts({k: sv.getncattr(k) for k in sv.ncattrs()})
        dv[:] = sv[:]
        print(f"  lead_month already canonical — copied.")

    # ── 4. tp: load ALL into RAM, transpose, write in one pass ───────────────
    print("\nStep 4/5 — tp transpose …")
    if "tp" in src.variables:
        sv = src.variables["tp"]
        canonical = ("time", "number", "latitude", "longitude")
        if sv.dimensions != canonical:
            t0 = time.time()
            gb = sv.size * 4 / 1e9
            print(f"  Loading tp {sv.dimensions} — {gb:.2f} GB into RAM …")
            tp_raw = sv[:]                             # e.g. (num, lat, lon, time)
            print(f"  Loaded in {time.time()-t0:.0f}s.")
            perm = [list(sv.dimensions).index(d) for d in canonical]
            t0 = time.time()
            print(f"  Transposing axes {perm} …")
            tp_fixed = np.transpose(tp_raw, perm)      # (time, num, lat, lon)
            print(f"  Transposed in {time.time()-t0:.0f}s.")
            del tp_raw
            t0 = time.time()
            dv = dst.createVariable(
                "tp", "f4", canonical,
                zlib=True, complevel=4, shuffle=True,
                chunksizes=(_T_CHUNK, n_number, n_lat, n_lon),
            )
            dv.setncatts({k: sv.getncattr(k) for k in sv.ncattrs()})
            dv[:] = tp_fixed
            del tp_fixed
            print(f"  Written in {time.time()-t0:.0f}s.")
        else:
            print(f"  tp already canonical — streaming copy …")
            dv = dst.createVariable(
                "tp", "f4", canonical,
                zlib=True, complevel=4, shuffle=True,
                chunksizes=(_T_CHUNK, n_number, n_lat, n_lon),
            )
            dv.setncatts({k: sv.getncattr(k) for k in sv.ncattrs()})
            for ts in range(0, n_time, _T_CHUNK):
                dv[ts:ts+_T_CHUNK] = sv[ts:ts+_T_CHUNK]

    # ── 5. stream-copy all remaining data variables ──────────────────────────
    print("\nStep 5/5 — streaming copy of remaining variables …")
    skip = {"time", "number", "latitude", "longitude", "cci_class",
            "tp", "lead_time", "lead_month"}
    for vname in src.variables:
        if vname in skip:
            continue
        sv = src.variables[vname]
        dims = sv.dimensions

        if dims == ("time", "number", "latitude", "longitude"):
            chunks = (_T_CHUNK, n_number, n_lat, n_lon)
        elif dims == ("time", "cci_class", "latitude", "longitude"):
            chunks = (_T_CHUNK, n_cci, n_lat, n_lon)
        elif dims == ("time", "latitude", "longitude"):
            chunks = (_T_CHUNK, n_lat, n_lon)
        else:
            chunks = None

        kwargs = dict(zlib=True, complevel=4, shuffle=True)
        if chunks:
            kwargs["chunksizes"] = chunks
        dv = dst.createVariable(vname, sv.dtype, dims, **kwargs)
        dv.setncatts({k: sv.getncattr(k) for k in sv.ncattrs()})

        t0 = time.time()
        if dims and dims[0] == "time":
            for ts in range(0, n_time, _T_CHUNK):
                dv[ts:ts+_T_CHUNK] = sv[ts:ts+_T_CHUNK]
        else:
            dv[:] = sv[:]
        print(f"  {vname:20s}  {dims}  "
              f"{sv.size*4/1e9:.2f} GB  {time.time()-t0:.0f}s")

    src.close()
    dst.close()

    out_gb = tmp_path.stat().st_size / 1e9
    print(f"\n  Output: {out_gb:.2f} GB")
    print(f"  Renaming to {nc_path.name} …")
    os.rename(tmp_path, nc_path)
    print(f"  Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check / fix seasonal NetCDF structure for SeasonalDataLoader."
    )
    parser.add_argument("--nc_path", required=True,
                        help="Path to the seasonal NetCDF file.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Inspect only; do not write any output.")
    args = parser.parse_args()

    nc_path = Path(args.nc_path).expanduser().resolve()
    if not nc_path.exists():
        print(f"ERROR: file not found: {nc_path}", file=sys.stderr)
        sys.exit(1)

    status = inspect(nc_path)
    if not status["needs_work"]:
        sys.exit(0)

    if args.dry_run:
        print("  DRY-RUN — exiting without writing.\n")
        sys.exit(0)

    t_total = time.time()
    preprocess(nc_path)
    print(f"\nTotal time: {time.time()-t_total:.0f}s")

    print("\nVerifying output …")
    final = inspect(nc_path)
    if final["needs_work"]:
        print("ERROR: file still has structural issues!", file=sys.stderr)
        sys.exit(1)
    print("Verification passed.\n")


if __name__ == "__main__":
    main()
