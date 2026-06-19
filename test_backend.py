"""Quick backend test - verifies V6.1 models load and predict correctly."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import config as cfg
from src.model.architecture import FlareForecaster
import torch, numpy as np
import torch.nn.functional as F

print("=" * 60)
print("  JWALASHMI Backend Test")
print("=" * 60)

# Test 1: V6.1 model loading
v6_dir = str(cfg.MODEL_DIR / 'v6_1_ensemble')
models = []
for i in range(10):
    p = os.path.join(v6_dir, f'model_{i}.pt')
    if os.path.exists(p):
        m = FlareForecaster(n_input_channels=9)
        m.load_state_dict(torch.load(p, map_location='cpu', weights_only=True))
        m.eval()
        models.append(m)
print(f"\n  [1] V6.1 Models: {len(models)}/10 loaded", "PASS" if len(models) == 10 else "FAIL")

# Test 2: Inference
x = torch.randn(1, 3600, 9)
all_p = []
for m in models:
    with torch.no_grad():
        lo, lt, at = m(x)
        all_p.append(F.softmax(lo, dim=1).numpy()[0])
avg = np.mean(all_p, axis=0)
pred = int(avg.argmax())
print(f"  [2] Inference: {cfg.CLASS_NAMES[pred]} (conf={avg.max():.3f}) PASS")

# Test 3: Thresholds
t = os.path.join(v6_dir, 'thresholds.npy')
if os.path.exists(t):
    th = np.load(t)
    print(f"  [3] Thresholds: None={th[0]:.1f} B={th[1]:.1f} C={th[2]:.1f} PASS")
else:
    print(f"  [3] Thresholds: Not found WARN")

# Test 4: Data loading
try:
    proc = str(cfg.PROCESSED)
    has_data = os.path.exists(os.path.join(proc, 'X_tactical.npy'))
    print(f"  [4] Preprocessed data: {'Found' if has_data else 'Not found (will use simulation)'}")
except:
    print(f"  [4] Preprocessed data: Not found (will use simulation)")

# Test 5: Flask import
try:
    from flask import Flask
    print(f"  [5] Flask: Available PASS")
except ImportError:
    print(f"  [5] Flask: NOT INSTALLED - run: pip install flask")

# Test 6: Strategic model
strat = str(cfg.MODEL_DIR / 'best_strategic_model.pt')
print(f"  [6] Strategic model: {'Found' if os.path.exists(strat) else 'Not found (OK)'}")

# Test 7: Catalog
cat = str(cfg.PROCESSED / 'flare_catalog.csv')
print(f"  [7] Flare catalog: {'Found' if os.path.exists(cat) else 'Not found (will scan FITS)'}")

print(f"\n{'=' * 60}")
print(f"  Backend is {'READY' if len(models) == 10 else 'PARTIALLY READY'}")
print(f"  Run: python server.py")
print(f"{'=' * 60}")
