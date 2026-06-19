"""Create a minimal zip for Colab training (no FITS files)."""
import os, zipfile, sys
sys.path.insert(0, ".")

zip_path = "JWALASHMI_colab_v62.zip"
root = os.path.dirname(os.path.abspath(__file__))

files_to_include = [
    # Config
    "config.py",
    # Training script
    "train_v6_2_colab.py",
    # V5 training (for oversample_minorities)
    "run_v5_max_accuracy.py",
    # Preprocessed data
    "data/processed/X_tactical.npy",
    "data/processed/y_tactical.npy",
    "data/processed/lead_times.npy",
    "data/processed/feature_mean.npy",
    "data/processed/feature_std.npy",
    # Model code
    "src/__init__.py",
    "src/model/__init__.py",
    "src/model/architecture.py",
    "src/model/augmentation.py",
    "src/model/ensemble.py",
    "src/model/loss.py",
    # Feature code (for reference)
    "src/features/__init__.py",
    "src/features/physics_features.py",
    "src/features/windowing.py",
    # Nowcasting
    "src/nowcasting/__init__.py",
    "src/nowcasting/detector.py",
    "src/nowcasting/classifier.py",
    "src/nowcasting/catalog.py",
    # Data
    "src/data/__init__.py",
    "src/data/fits_loader.py",
]

print("Creating %s..." % zip_path)
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in files_to_include:
        full = os.path.join(root, f)
        if os.path.exists(full):
            zf.write(full, f)
            size = os.path.getsize(full)
            print("  + %s (%.1f MB)" % (f, size / 1024 / 1024))
        else:
            print("  SKIP %s (not found)" % f)

    # Create empty dirs
    for d in ["models", "models/v6_2_ensemble", "data/processed"]:
        zf.writestr(d + "/", "")

final_size = os.path.getsize(zip_path) / 1024 / 1024
print("\nDone! %s = %.1f MB" % (zip_path, final_size))
print("Upload this to Google Colab and run train_v6_2_colab.py")
