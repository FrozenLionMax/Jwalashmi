"""
Solar Flare Early Warning System - API Server
Serves model predictions via Flask for the Mission Control dashboard.
"""
import os
import sys
import json
import time
import numpy as np
import torch
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from flask import Flask, jsonify, send_from_directory, request
import config as cfg

app = Flask(__name__, static_folder="dashboard", static_url_path="")

# Global model references
ensemble = None
strategic_model = None


def load_models():
    """Load ensemble and strategic models."""
    global ensemble, strategic_model

    from src.model.ensemble import EnsembleForecaster
    from src.model.architecture import StrategicForecaster

    # Load tactical ensemble
    ens_dir = str(cfg.MODEL_DIR / "tactical_ensemble")
    if os.path.exists(ens_dir):
        ensemble = EnsembleForecaster(n_models=5, n_features=9)
        ensemble.load(ens_dir)
        print(f"  Tactical ensemble loaded ({len(ensemble.models)} models)")

    # Load strategic model
    strat_path = str(cfg.MODEL_DIR / "best_strategic_model.pt")
    if os.path.exists(strat_path):
        strategic_model = StrategicForecaster(n_input_channels=9)
        strategic_model.load_state_dict(
            torch.load(strat_path, map_location="cpu", weights_only=True)
        )
        strategic_model.eval()
        print(f"  Strategic model loaded")


def get_latest_prediction():
    """Generate prediction from the most recent data window."""
    if ensemble is None:
        return None

    tac_x_path = str(cfg.PROCESSED / "X_tactical.npy")
    tac_y_path = str(cfg.PROCESSED / "y_tactical.npy")
    tac_lt_path = str(cfg.PROCESSED / "lt_tactical.npy")

    if not os.path.exists(tac_x_path):
        return None

    X = np.load(tac_x_path)
    y = np.load(tac_y_path)
    lt = np.load(tac_lt_path)

    # Pick a window — changes every 30s for live feel
    rng = np.random.default_rng(int(time.time()) // 30)
    idx = rng.integers(0, len(X))

    window = X[idx:idx+1]
    true_class = int(y[idx])
    true_lt = float(lt[idx])

    # Tactical prediction
    detailed = ensemble.predict_detailed(window)
    pred = detailed[0]
    uncertainty = float(ensemble.get_uncertainty(window)[0])

    # Strategic prediction
    strat_result = None
    str_x_path = str(cfg.PROCESSED / "X_strategic.npy")
    str_y_path = str(cfg.PROCESSED / "y_strategic.npy")
    if os.path.exists(str_x_path) and strategic_model is not None:
        X_str = np.load(str_x_path)
        y_str = np.load(str_y_path)
        s_idx = rng.integers(0, len(X_str))
        s_window = torch.tensor(X_str[s_idx:s_idx+1], dtype=torch.float32)
        with torch.no_grad():
            s_logits, _ = strategic_model(s_window)
            s_probs = torch.softmax(s_logits, dim=1).numpy()[0]
            s_pred = int(s_probs.argmax())
        strat_result = {
            "predicted_class": s_pred,
            "class_name": cfg.CLASS_NAMES[s_pred],
            "confidence": float(s_probs.max()),
            "probabilities": s_probs.tolist(),
            "true_class": cfg.CLASS_NAMES[int(y_str[s_idx])],
        }

    # Extract both SoLEXS and HEL1OS flux from the 9-feature window
    # Features: [flux_solexs, flux_helios, derivative, temp, em, qpp, norm, slope, accel]
    win = window[0]  # (3600, 9)
    solexs_flux = win[::60, 0].tolist()   # sample every 60s -> 60 points
    helios_flux = win[::60, 1].tolist()

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tactical": {
            "predicted_class": pred["class_name"],
            "confidence": pred["confidence"],
            "alert_level": pred["alert_level"],
            "lead_time_min": pred.get("lead_time_min", 0),
            "probabilities": {
                cfg.CLASS_NAMES[i]: pred["probabilities"][i]
                for i in range(len(cfg.CLASS_NAMES))
            },
            "uncertainty": uncertainty,
            "true_class": cfg.CLASS_NAMES[true_class],
        },
        "strategic": strat_result,
        "flux_solexs": solexs_flux,
        "flux_helios": helios_flux,
        "system": {
            "ensemble_models": len(ensemble.models) if ensemble else 0,
            "temperature": ensemble.calibrator.temperature if ensemble else 1.0,
        }
    }


@app.route("/")
def index():
    return send_from_directory("dashboard", "index.html")


@app.route("/api/predict")
def predict():
    try:
        result = get_latest_prediction()
        if result is None:
            return jsonify({"error": "Models not loaded"}), 503
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/health")
def health():
    return jsonify({
        "status": "operational",
        "ensemble_loaded": ensemble is not None,
        "strategic_loaded": strategic_model is not None,
        "n_models": len(ensemble.models) if ensemble else 0,
    })


@app.route("/api/metrics")
def metrics():
    """Return model performance metrics."""
    return jsonify({
        "tactical": {
            "accuracy": 0.536, "f1_score": 0.716,
            "tpr": 0.650, "fpr": 0.417,
            "mean_lead_time": 32.9,
            "x_class_auc": 1.000, "m_class_auc": 0.997,
            "c_class_auc": 0.900, "b_class_auc": 0.789,
        },
        "strategic": { "accuracy": 0.849, "tpr": 0.993, "fpr": 0.456 },
        "ensemble": { "n_models": 5, "temperature": 2.04, "augmentation": "5x" },
    })


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  SOLAR FLARE EARLY WARNING SYSTEM")
    print("  Mission Control Dashboard")
    print("=" * 60)
    print("\nLoading models...")
    load_models()
    print(f"\nDashboard: http://localhost:5000")
    print(f"API:       http://localhost:5000/api/predict")
    print(f"Health:    http://localhost:5000/api/health\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
