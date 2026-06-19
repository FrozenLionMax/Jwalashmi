"""Check HEL1OS FITS format."""
from astropy.io import fits
import glob

# Find HEL1OS files
files = glob.glob(r"c:\Users\Acer\Desktop\ISRO\Helios\extracted\**\lightcurve_*.fits", recursive=True)
print(f"Found {len(files)} HEL1OS lightcurve files")

if files:
    f = files[0]
    print(f"\nSample: {f}")
    hdu = fits.open(f)
    print(f"HDUs: {len(hdu)}")
    for i, h in enumerate(hdu):
        print(f"  HDU {i}: {h.name}")
        if hasattr(h, 'columns') and h.columns is not None:
            print(f"    Columns: {h.columns.names}")
            print(f"    Rows: {len(h.data)}")
            # Show first few values of each column
            for col in h.columns.names[:10]:
                vals = h.data[col]
                print(f"    {col}: shape={vals.shape if hasattr(vals,'shape') else 'scalar'} sample={vals[:3]}")
    hdu.close()

    # Check how many dates
    dates = set()
    for f2 in files:
        parts = f2.replace("\\", "/").split("/")
        for p in parts:
            if len(p) == 2 and p.isdigit():
                continue
            if len(p) == 4 and p.startswith("20"):
                year = p
        dates.add(f2.split("extracted")[1][:12])
    print(f"\nUnique date-paths: {len(dates)}")
    for d in sorted(dates)[:10]:
        print(f"  {d}")
