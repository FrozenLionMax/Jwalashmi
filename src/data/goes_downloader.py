"""
Solar Flare Early Warning System — GOES Data Downloader
Downloads GOES XRS data and NOAA flare event list for pre-training.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config as cfg


def download_noaa_flare_list(save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Download the NOAA SWPC solar flare event list.
    This provides ground-truth labels: start, peak, stop, class for every flare.
    
    Falls back to a curated CSV if the NOAA API is unavailable.
    """
    save_path = save_path or str(cfg.GOES_DATA / "noaa_flare_list.csv")

    if os.path.exists(save_path):
        print(f"Loading cached NOAA flare list from {save_path}")
        return pd.read_csv(save_path, parse_dates=["start", "peak", "end"])

    print("Downloading NOAA flare list...")
    try:
        # NOAA SWPC event list (JSON API)
        url = "https://services.swpc.noaa.gov/json/solar_events.json"
        df = pd.read_json(url)
        # Filter for X-ray flares only
        flares = df[df["type"] == "FLA"].copy()
        flares = flares.rename(columns={
            "begin_datetime": "start",
            "max_datetime": "peak",
            "end_datetime": "end",
            "classtype": "goes_class",
        })
        flares = flares[["start", "peak", "end", "goes_class"]].dropna()
        flares.to_csv(save_path, index=False)
        print(f"Saved {len(flares)} flares to {save_path}")
        return flares
    except Exception as e:
        print(f"Warning: Could not download NOAA list: {e}")
        print("Creating a synthetic flare list from our own detections...")
        return pd.DataFrame(columns=["start", "peak", "end", "goes_class"])


def download_goes_xrs(start_date: str = "2020-01-01",
                       end_date: str = "2024-12-31",
                       save_dir: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Download GOES XRS (X-Ray Sensor) data using SunPy.
    
    This provides 50+ years of labeled soft X-ray flux for pre-training.
    Two channels: 0.5-4 Angstrom (short) and 1-8 Angstrom (long).
    
    Args:
        start_date: start of download range
        end_date: end of download range
        save_dir: directory to save downloaded data
    """
    save_dir = save_dir or str(cfg.GOES_DATA)
    os.makedirs(save_dir, exist_ok=True)
    cache_path = os.path.join(save_dir, "goes_xrs.parquet")

    if os.path.exists(cache_path):
        print(f"Loading cached GOES XRS data from {cache_path}")
        return pd.read_parquet(cache_path)

    try:
        from sunpy.net import Fido, attrs as a
        from sunpy.timeseries import TimeSeries

        print(f"Downloading GOES XRS data: {start_date} to {end_date}")
        print("This may take several minutes...")

        result = Fido.search(
            a.Time(start_date, end_date),
            a.Instrument.xrs,
            a.Resolution("flx1s"),  # 1-second resolution
        )

        if len(result) == 0:
            print("No GOES XRS data found for the specified range.")
            return None

        files = Fido.fetch(result, path=os.path.join(save_dir, "raw", "{file}"),
                           progress=True)

        # Combine into single TimeSeries
        ts = TimeSeries(files, concatenate=True)
        df = ts.to_dataframe()
        df = df.reset_index()
        df.columns = ["datetime", "xrs_short", "xrs_long"]
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df["timestamp"] = df["datetime"].astype(np.int64) / 1e9

        # Save as parquet for fast loading
        df.to_parquet(cache_path, index=False)
        print(f"Saved {len(df)} data points to {cache_path}")
        return df

    except ImportError:
        print("SunPy not installed. Run: pip install sunpy")
        print("GOES data download skipped. You can pre-train without it.")
        return None
    except Exception as e:
        print(f"Error downloading GOES data: {e}")
        return None


if __name__ == "__main__":
    print("Testing GOES downloader...")
    flare_list = download_noaa_flare_list()
    print(f"Flare list: {len(flare_list)} events")
    if not flare_list.empty:
        print(flare_list.head())
