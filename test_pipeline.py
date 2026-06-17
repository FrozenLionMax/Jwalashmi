"""
End-to-end test of the Solar Flare Early Warning System pipeline.
Tests: data loading → nowcasting → feature engineering → model.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

def test_data_loader():
    print("=" * 60)
    print("  TEST 1: Data Loader")
    print("=" * 60)

    from src.data.fits_loader import (find_solexs_files, find_hel1os_files,
                                       load_solexs_lightcurve, load_hel1os_lightcurve)

    solexs_files = find_solexs_files()
    print(f"\nSoLEXS: Found {len(solexs_files)} days")
    for f in solexs_files[:3]:
        print(f"  {f['date']}")
    if len(solexs_files) > 3:
        print(f"  ... and {len(solexs_files) - 3} more")

    df = load_solexs_lightcurve(solexs_files[0]["lc_path"])
    print(f"\nSample SoLEXS ({solexs_files[0]['date']}):")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    date_range = f"{df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}"
    print(f"  Time: {date_range}")
    print(f"  Counts: min={df['counts'].min():.0f}, max={df['counts'].max():.0f}")

    hel1os_files = find_hel1os_files()
    print(f"\nHEL1OS: Found {len(hel1os_files)} files")
    for f in hel1os_files[:5]:
        print(f"  {f['date']} / {f['detector']}")

    if hel1os_files:
        df_h = load_hel1os_lightcurve(hel1os_files[0]["path"])
        print(f"\nSample HEL1OS ({hel1os_files[0]['date']}):")
        print(f"  Shape: {df_h.shape}")
        ctr_cols = [c for c in df_h.columns if c.startswith("ctr_")]
        print(f"  CTR columns: {ctr_cols}")
        print(f"  Detector: {df_h['detector'].iloc[0]}")

    print("\n[PASS] Data loader works!")
    return df, solexs_files


def test_nowcasting(df):
    print("\n" + "=" * 60)
    print("  TEST 2: Nowcasting Detector")
    print("=" * 60)

    from src.nowcasting.detector import detect_flares, flares_to_dataframe

    flares = detect_flares(df, instrument="solexs")
    print(f"\nDetected {len(flares)} flares:")
    for f in flares[:10]:
        print(f"  {f.peak_dt} | Class {f.estimated_class} | "
              f"Peak: {f.peak_counts:.0f} cts/s | "
              f"Duration: {f.duration:.0f}s | "
              f"Confidence: {f.confidence:.2f}")

    if flares:
        catalog_df = flares_to_dataframe(flares)
        print(f"\nCatalog DataFrame shape: {catalog_df.shape}")
        print(f"Classes found: {catalog_df['estimated_class'].value_counts().to_dict()}")

    print("\n[PASS] Nowcasting detector works!")
    return flares


def test_features(df):
    print("\n" + "=" * 60)
    print("  TEST 3: Feature Engineering")
    print("=" * 60)

    from src.features.physics_features import compute_all_features, get_feature_columns

    df_feat = compute_all_features(df)
    feat_cols = get_feature_columns(df_feat)
    print(f"\nComputed {len(feat_cols)} features:")
    for c in feat_cols:
        vals = df_feat[c].values
        print(f"  {c:30s}  min={np.nanmin(vals):12.4f}  "
              f"max={np.nanmax(vals):12.4f}  mean={np.nanmean(vals):12.4f}")

    print("\n[PASS] Feature engineering works!")
    return df_feat, feat_cols


def test_windowing(df_feat, feat_cols, flares):
    print("\n" + "=" * 60)
    print("  TEST 4: Sliding Windows")
    print("=" * 60)

    from src.features.windowing import create_windows, print_window_stats
    from src.nowcasting.detector import flares_to_dataframe

    catalog = flares_to_dataframe(flares) if flares else None

    X, y, metadata = create_windows(
        df_feat, feat_cols, flare_catalog=catalog,
        window_size=1800,  # 30 min for faster test
        stride=600,        # 10 min stride
    )

    print(f"\nWindows created:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print_window_stats(y, metadata)

    print("\n[PASS] Windowing works!")
    return X, y


def test_model(X, y):
    print("\n" + "=" * 60)
    print("  TEST 5: Model Architecture")
    print("=" * 60)

    from src.model.architecture import FlareForecaster, print_model_summary
    import torch

    n_features = X.shape[2]
    window_size = X.shape[1]

    model = FlareForecaster(n_input_channels=n_features)
    params = model.count_parameters()
    print(f"\nModel created:")
    print(f"  Input: (batch, {window_size}, {n_features})")
    print(f"  Parameters: {params['total']:,}")

    # Test forward pass
    x_test = torch.randn(2, window_size, n_features)
    logits, lead_time, attn = model(x_test)
    print(f"\nForward pass:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Lead time shape: {lead_time.shape}")
    print(f"  Attention shape: {attn.shape}")

    # Test freeze/unfreeze
    model.freeze_cnn()
    params_frozen = model.count_parameters()
    print(f"  After freeze: {params_frozen['trainable']:,} trainable")

    model.unfreeze_cnn()
    params_unfrozen = model.count_parameters()
    print(f"  After unfreeze: {params_unfrozen['trainable']:,} trainable")

    print("\n[PASS] Model architecture works!")
    return model


def test_loss():
    print("\n" + "=" * 60)
    print("  TEST 6: Loss Function")
    print("=" * 60)

    from src.model.architecture import FlareForecasterLoss
    import torch

    criterion = FlareForecasterLoss()

    # Fake batch
    logits = torch.randn(4, 5)
    lead_pred = torch.rand(4, 1) * 30
    targets = torch.tensor([0, 2, 3, 0])
    lead_true = torch.tensor([0, 15, 25, 0], dtype=torch.float32)

    losses = criterion(logits, lead_pred, targets, lead_true)
    print(f"\nLoss components:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")

    print("\n[PASS] Loss function works!")


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  SOLAR FLARE EARLY WARNING SYSTEM — END-TO-END TEST")
    print("#" * 60)

    df, files = test_data_loader()
    flares = test_nowcasting(df)
    df_feat, feat_cols = test_features(df)
    X, y = test_windowing(df_feat, feat_cols, flares)
    model = test_model(X, y)
    test_loss()

    print("\n" + "#" * 60)
    print("  ALL TESTS PASSED!")
    print("#" * 60)
    print(f"\n  Summary:")
    print(f"    Data:     {len(files)} SoLEXS days loaded")
    print(f"    Flares:   {len(flares)} detected")
    print(f"    Features: {len(feat_cols)} physics features")
    print(f"    Windows:  {X.shape[0]} training windows")
    print(f"    Model:    {model.count_parameters()['total']:,} parameters")
    print(f"\n  System is READY for training and deployment.")
