"""
JWALASHMI — HEL1OS Data Download Guide
=======================================
PRADAN (https://pradan.issdc.gov.in/al1) requires manual login.
This script cannot auto-download, but it helps you know EXACTLY what to download.

After you download the zip files, run:
    python download_helios_guide.py --extract
to auto-extract and verify the data.
"""
import os
import sys
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The SoLEXS dates we need HEL1OS data for
SOLEXS_DATES = [
    "2024-02-22", "2024-02-23",
    "2024-05-05", "2024-05-06", "2024-05-07", "2024-05-08",
    "2024-05-09", "2024-05-10", "2024-05-11",
    "2024-05-22", "2024-05-23", "2024-05-24", "2024-05-25",
    "2024-05-26", "2024-05-27", "2024-05-28", "2024-05-29",
    "2024-05-30", "2024-05-31",
    "2024-08-04",
    "2024-10-03",  # Already have
    "2024-12-29", "2024-12-30", "2024-12-31",
    "2025-11-05",  # Already have
]

ALREADY_HAVE = ["2024-10-03", "2025-11-05", "2025-11-11", "2026-02-01", "2026-06-10"]

# Major solar flare dates during Aditya-L1 (HIGHEST PRIORITY)
FLARE_DATES = {
    "2024-02-22": "X6.3 flare (AR3590) — Strongest of Solar Cycle 25 at that time",
    "2024-02-23": "M4.8 flare aftermath",
    "2024-05-08": "X1.0 flare (AR3664) — Start of historic May storm",
    "2024-05-09": "X2.2 + X1.2 flares — Double X-class day",
    "2024-05-10": "X3.9 flare — G5 geomagnetic storm trigger",
    "2024-05-11": "X5.8 flare — Largest of the storm, aurora worldwide",
    "2024-05-14": "X8.7 flare — Largest of Solar Cycle 25",
    "2024-08-04": "M-class flare day",
    "2024-10-03": "Active region flares (ALREADY HAVE)",
}


def print_download_guide():
    print("=" * 70)
    print("  HEL1OS DATA DOWNLOAD GUIDE")
    print("  Go to: https://pradan.issdc.gov.in/al1/protected/payload.xhtml")
    print("=" * 70)

    needed = [d for d in SOLEXS_DATES if d not in ALREADY_HAVE]
    print(f"\n  You already have HEL1OS for: {len(ALREADY_HAVE)} dates")
    print(f"  You NEED HEL1OS for: {len(needed)} more dates\n")

    print("  PRIORITY 1 — FLARE DATES (download these FIRST):")
    print("  " + "-" * 60)
    for date, desc in FLARE_DATES.items():
        status = "HAVE" if date in ALREADY_HAVE else "NEED"
        marker = "[x]" if status == "HAVE" else "[ ]"
        print(f"    {marker} {date}  {desc}")

    print(f"\n  PRIORITY 2 — ALL MATCHING SOLEXS DATES:")
    print("  " + "-" * 60)
    for date in SOLEXS_DATES:
        status = "HAVE" if date in ALREADY_HAVE else "NEED"
        marker = "[x]" if status == "HAVE" else "[ ]"
        flare = FLARE_DATES.get(date, "")
        if flare:
            flare = f" *** {flare}"
        print(f"    {marker} {date}{flare}")

    print(f"""
  STEPS TO DOWNLOAD:
  ------------------
  1. Go to https://pradan.issdc.gov.in/al1
  2. Click "Login/Signup" (top right)
  3. Register or login with your account
  4. Click "Browse and Download"
  5. Select payload: HEL1OS
  6. Select Level: Level-1
  7. For EACH date above:
     - Set Start Date and End Date to the same day
     - Click Search
     - Select ALL files (lightcurve_cdte1.fits, cdte2, czt1, czt2)
     - Click "Bulk Download"
  8. Save zip files to: Helios/ folder in your project
  9. Run: python download_helios_guide.py --extract

  MINIMUM needed for hackathon: Download May 8-11, 2024
  (4 dates with X-class flares = most impactful for judges)
""")


def extract_and_verify():
    """Extract downloaded HEL1OS zips and verify."""
    print("\n  Extracting HEL1OS data...")
    from src.data.extract_all import extract_hel1os_zips
    extract_hel1os_zips()

    # Verify overlap
    print("\n  Verifying SoLEXS-HEL1OS overlap...")
    from src.data.fits_loader import find_solexs_files, find_hel1os_files
    s_dates = set(f["date"] for f in find_solexs_files())
    h_dates = set(f["date"] for f in find_hel1os_files())
    overlap = s_dates & h_dates

    print(f"\n  SoLEXS dates: {len(s_dates)}")
    print(f"  HEL1OS dates: {len(h_dates)}")
    print(f"  OVERLAP:      {len(overlap)} dates")
    print(f"  Overlapping:  {sorted(overlap)}")

    if len(overlap) >= 5:
        print("\n  GOOD! You have enough overlap for meaningful HEL1OS features.")
    elif len(overlap) >= 2:
        print("\n  OK but download more HEL1OS dates for better results.")
    else:
        print("\n  WARNING: Need more HEL1OS data. Download from PRADAN.")

    missing = s_dates - h_dates
    if missing:
        print(f"\n  Still missing HEL1OS for: {sorted(missing)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HEL1OS download guide")
    parser.add_argument("--extract", action="store_true", help="Extract downloaded zips")
    args = parser.parse_args()

    if args.extract:
        extract_and_verify()
    else:
        print_download_guide()
