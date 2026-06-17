"""
Solar Flare Early Warning System — HEL1OS Data Extractor
Extracts only light curve files from HEL1OS zip archives to save disk space.
"""
import os
import sys
import zipfile
import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config as cfg


def extract_hel1os_zips(zip_dir: str = None, output_dir: str = None,
                         lightcurves_only: bool = True):
    """
    Extract all HEL1OS zip files, handling nested zips.
    Only extracts light curves and GTI files to save disk space.
    """
    zip_dir = zip_dir or str(cfg.HEL1OS_ZIPS)
    output_dir = output_dir or str(cfg.HEL1OS_RAW)
    os.makedirs(output_dir, exist_ok=True)

    # Find all zip files
    zips = sorted(glob.glob(os.path.join(zip_dir, "*.zip")))
    print(f"Found {len(zips)} zip files in {zip_dir}")

    for zip_path in zips:
        name = os.path.basename(zip_path)
        print(f"\nProcessing: {name}")

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                contents = zf.namelist()

                # Check if contains nested zips
                inner_zips = [f for f in contents if f.endswith(".zip")]

                if inner_zips:
                    # Extract inner zips temporarily, then extract their contents
                    for inner in inner_zips:
                        print(f"  Nested zip: {inner}")

                        # Check if already extracted
                        inner_base = inner.replace(".zip", "")
                        check_pattern = os.path.join(output_dir, "**",
                                                      "lightcurve_cdte1.fits")
                        # Simple check: see if any dir matches the date
                        date_part = inner.split("_")[1] if "_" in inner else ""
                        if date_part:
                            year = date_part[:4]
                            month = date_part[4:6]
                            day = date_part[6:8]
                            check_dir = os.path.join(output_dir, year, month, day)
                            if os.path.exists(check_dir):
                                existing = glob.glob(os.path.join(check_dir,
                                                                   "**", "lightcurve_*.fits"),
                                                      recursive=True)
                                if existing:
                                    print(f"    Already extracted ({len(existing)} files)")
                                    continue

                        # Extract inner zip to temp location
                        zf.extract(inner, output_dir)
                        inner_path = os.path.join(output_dir, inner)

                        # Now extract from inner zip
                        try:
                            with zipfile.ZipFile(inner_path, "r") as izf:
                                for f in izf.namelist():
                                    if lightcurves_only:
                                        if ("lightcurve" in f or "gti" in f
                                                or f.endswith("/")):
                                            izf.extract(f, output_dir)
                                            if not f.endswith("/"):
                                                size = izf.getinfo(f).file_size / 1024
                                                print(f"    Extracted: {os.path.basename(f)} "
                                                      f"({size:.0f} KB)")
                                    else:
                                        izf.extract(f, output_dir)
                        finally:
                            os.remove(inner_path)
                else:
                    # Direct extraction (single-level zip like HLS_*.zip)
                    for f in contents:
                        if lightcurves_only:
                            if ("lightcurve" in f or "gti" in f or f.endswith("/")):
                                zf.extract(f, output_dir)
                                if not f.endswith("/"):
                                    size = zf.getinfo(f).file_size / 1024
                                    print(f"  Extracted: {os.path.basename(f)} "
                                          f"({size:.0f} KB)")
                        else:
                            zf.extract(f, output_dir)

        except zipfile.BadZipFile:
            print(f"  ERROR: Bad zip file (possibly still downloading?)")
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nDone. Checking extracted files...")
    all_lc = glob.glob(os.path.join(output_dir, "**", "lightcurve_*.fits"),
                        recursive=True)
    print(f"Total light curve files: {len(all_lc)}")

    # Group by date
    dates = set()
    for f in all_lc:
        parts = Path(f).parts
        for j, p in enumerate(parts):
            if p.isdigit() and len(p) == 4 and int(p) > 2000:
                try:
                    dates.add(f"{parts[j]}-{parts[j+1]}-{parts[j+2]}")
                except IndexError:
                    pass
                break

    print(f"Dates with data: {sorted(dates)}")


if __name__ == "__main__":
    extract_hel1os_zips()
