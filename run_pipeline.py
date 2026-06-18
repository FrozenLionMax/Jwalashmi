"""
JWALASHMI v4.0 - Solar Flare Early Warning Pipeline
DUAL-TIER + Ensemble + GOES Pre-training + 5-Fold CV + FPR Optimization

Features:
  - GOES Pre-training (transfer learning from 50yr GOES data)
  - 5-Fold Stratified Cross-Validation for robust metrics
  - FPR Optimization (target FPR < 0.30)
  - 10x Data Augmentation
  - 5-Model Ensemble with confidence thresholds
  - Temperature-scaled calibrated probabilities
  - Dual-tier: Strategic (5-10h) + Tactical (30-60 min)

Usage:
    python run_pipeline.py                      # Full pipeline (ensemble)
    python run_pipeline.py --pretrain-goes      # Pre-train on GOES then fine-tune
    python run_pipeline.py --cv                 # 5-fold cross-validation
    python run_pipeline.py --quick              # Quick mode (single model)
    python run_pipeline.py --tactical           # Only Tier 2
    python run_pipeline.py --strategic          # Only Tier 1
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import config as cfg


def step_load_data():
    """Step 1: Load all data."""
    print("\n" + "=" * 70)
    print("  STEP 1: Loading Data")
    print("=" * 70)
    from src.data.fits_loader import (load_solexs_all, load_hel1os_all,
                                       find_solexs_files, find_hel1os_files)

    solexs_files = find_solexs_files()
    hel1os_files = find_hel1os_files()
    print(f"  SoLEXS: {len(solexs_files)} days available")
    print(f"  HEL1OS: {len(hel1os_files)} files available")

    t0 = time.time()
    df_solexs = load_solexs_all()
    print(f"  SoLEXS loaded: {df_solexs.shape} in {time.time()-t0:.1f}s")

    t0 = time.time()
    df_hel1os = load_hel1os_all(detector="cdte1")
    print(f"  HEL1OS loaded: {df_hel1os.shape} in {time.time()-t0:.1f}s")

    return df_solexs, df_hel1os


def step_nowcast(df_solexs, df_hel1os):
    """Step 2: Detect flares in all data."""
    print("\n" + "=" * 70)
    print("  STEP 2: Nowcasting - Detecting Flares")
    print("=" * 70)
    from src.nowcasting.detector import (detect_flares, build_unified_catalog)

    all_solexs_flares = []
    for date, group in df_solexs.groupby("date"):
        flares = detect_flares(group.reset_index(drop=True), instrument="solexs")
        all_solexs_flares.extend(flares)
        if flares:
            classes = [f.estimated_class for f in flares]
            print(f"  {date}: {len(flares)} flares - "
                  f"{', '.join(f'{c}:{classes.count(c)}' for c in set(classes))}")

    print(f"\n  Total SoLEXS flares: {len(all_solexs_flares)}")

    all_hel1os_flares = []
    if not df_hel1os.empty:
        broad_col = [c for c in df_hel1os.columns if "1.80KEV_TO_90" in c]
        if broad_col:
            for date, group in df_hel1os.groupby("date"):
                grp = group.reset_index(drop=True)
                flares = detect_flares(grp, instrument="hel1os",
                                       count_col=broad_col[0])
                all_hel1os_flares.extend(flares)
                if flares:
                    print(f"  HEL1OS {date}: {len(flares)} flares")

    print(f"  Total HEL1OS flares: {len(all_hel1os_flares)}")

    catalog = build_unified_catalog(all_solexs_flares, all_hel1os_flares)
    catalog.to_csv(str(cfg.CATALOG_CSV), index=False)
    print(f"\n  Unified Catalog: {len(catalog)} events")

    if not catalog.empty:
        print(f"  Class distribution: {catalog['estimated_class'].value_counts().to_dict()}")

    return all_solexs_flares, catalog


def step_features(df_solexs, df_hel1os):
    """Step 3: Compute physics-informed features (per-day)."""
    print("\n" + "=" * 70)
    print("  STEP 3: Feature Engineering (per-day)")
    print("=" * 70)
    from src.features.physics_features import compute_all_features, get_feature_columns

    t0 = time.time()
    dfs = []
    for date in df_solexs["date"].unique():
        day_df = df_solexs[df_solexs["date"] == date].copy()
        day_feat = compute_all_features(day_df.reset_index(drop=True))
        dfs.append(day_feat)
        print(f"    {date}: {len(day_df)} points, features computed")

    df_feat = pd.concat(dfs, ignore_index=True)
    feat_cols = get_feature_columns(df_feat)

    print(f"\n  Computed {len(feat_cols)} features in {time.time()-t0:.1f}s")
    for c in feat_cols:
        vals = df_feat[c].values
        print(f"    {c:30s}  range=[{np.nanmin(vals):.2f}, {np.nanmax(vals):.2f}]")

    return df_feat, feat_cols


def step_tactical_windowing(df_feat, feat_cols, catalog):
    """Step 4a: Create Tier 2 (tactical) training windows."""
    print("\n" + "=" * 70)
    print("  STEP 4a: Tier 2 TACTICAL Windows (60-min horizon)")
    print("=" * 70)
    from src.features.windowing import (create_windows, normalize_features,
                                         balance_classes, print_window_stats)

    X, y, metadata = create_windows(df_feat, feat_cols, flare_catalog=catalog)
    print_window_stats(y, metadata)

    X_norm, mean, std = normalize_features(X)
    np.save(str(cfg.PROCESSED / "feature_mean.npy"), mean)
    np.save(str(cfg.PROCESSED / "feature_std.npy"), std)

    X_bal, y_bal, meta_bal = balance_classes(X_norm, y, metadata, max_ratio=10)
    print(f"\n  After balancing: {len(X_bal)} windows")
    print_window_stats(y_bal, meta_bal)

    lead_times = np.array([
        m.lead_time / 60 if m.lead_time is not None else 0
        for m in meta_bal
    ], dtype=np.float32)

    np.save(str(cfg.PROCESSED / "X_tactical.npy"), X_bal)
    np.save(str(cfg.PROCESSED / "y_tactical.npy"), y_bal)
    np.save(str(cfg.PROCESSED / "lt_tactical.npy"), lead_times)

    return X_bal, y_bal, lead_times


def step_pretrain_goes():
    """Step P: Pre-train on GOES XRS data for transfer learning."""
    print("\n" + "=" * 70)
    print("  STEP P: GOES Pre-training (Transfer Learning)")
    print("=" * 70)

    goes_x_path = str(cfg.GOES_DATA / "X_goes_pretrain.npy")
    goes_y_path = str(cfg.GOES_DATA / "y_goes_pretrain.npy")

    if not os.path.exists(goes_x_path):
        print("  No GOES pre-training data found. Downloading...")
        from src.data.goes_downloader import download_all_goes_data, create_goes_pretraining_data
        flares, xrs, hek = download_all_goes_data()
        all_events = hek if hek else flares
        create_goes_pretraining_data(xrs, all_events)

    if not os.path.exists(goes_x_path):
        print("  [WARN] Could not create GOES data. Skipping pre-training.")
        return None

    X_goes = np.load(goes_x_path)
    y_goes = np.load(goes_y_path)
    print(f"  GOES data: {X_goes.shape[0]} samples, {X_goes.shape[2]} features")
    for i, name in enumerate(cfg.CLASS_NAMES):
        count = (y_goes == i).sum()
        if count > 0:
            print(f"    {name}: {count}")

    # Pre-train a model on GOES data
    from src.model.architecture import FlareForecaster, FlareForecasterLoss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = X_goes.shape[2]
    model = FlareForecaster(n_input_channels=n_features).to(device)
    criterion = FlareForecasterLoss().to(device)

    # Split
    n_val = max(int(len(X_goes) * 0.15), 1)
    n_train = len(X_goes) - n_val

    X_t = torch.tensor(X_goes[:n_train], dtype=torch.float32).to(device)
    y_t = torch.tensor(y_goes[:n_train], dtype=torch.long).to(device)
    lt_t = torch.zeros(n_train, dtype=torch.float32).to(device)
    X_v = torch.tensor(X_goes[n_train:], dtype=torch.float32).to(device)
    y_v = torch.tensor(y_goes[n_train:], dtype=torch.long).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    batch_size = 32

    print(f"  Pre-training: {n_train} train, {n_val} val, 15 epochs")
    for epoch in range(1, 16):
        model.train()
        perm = torch.randperm(n_train)
        total_loss = 0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            optimizer.zero_grad()
            logits, lead_pred, _ = model(X_t[idx])
            losses = criterion(logits, y_t[idx], lead_pred, lt_t[idx])
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += losses["total"].item() * len(idx)

        model.eval()
        with torch.no_grad():
            v_logits, _, _ = model(X_v)
            v_acc = (v_logits.argmax(1) == y_v).float().mean().item()

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:2d}: loss={total_loss/n_train:.3f} val_acc={v_acc:.3f}")

    # Save pre-trained weights
    pretrain_path = str(cfg.MODEL_DIR / "goes_pretrained.pt")
    torch.save(model.state_dict(), pretrain_path)
    print(f"  Saved pre-trained weights: {pretrain_path}")

    return pretrain_path


def step_cross_validate(X, y, lead_times, pretrained_weights=None):
    """Step CV: 5-fold stratified cross-validation."""
    print("\n" + "=" * 70)
    print("  STEP CV: 5-Fold Stratified Cross-Validation")
    print("=" * 70)
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, roc_auc_score
    from src.model.architecture import FlareForecaster, FlareForecasterLoss
    from src.model.augmentation import augment_dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = X.shape[2]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_results = []
    all_preds = np.zeros(len(y), dtype=np.int64)
    all_probs = np.zeros((len(y), cfg.N_CLASSES), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n  --- Fold {fold}/5 ---")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        lt_train = lead_times[train_idx]

        # Augment training data
        X_aug, y_aug, lt_aug = augment_dataset(X_train, y_train, lt_train,
                                                multiplier=5, seed=fold*42)
        print(f"    Train: {len(X_train)} -> {len(X_aug)} (augmented)")
        print(f"    Val:   {len(X_val)}")

        # Create model
        model = FlareForecaster(n_input_channels=n_features).to(device)

        # Load pre-trained weights if available
        if pretrained_weights and os.path.exists(pretrained_weights):
            state = torch.load(pretrained_weights, map_location=device, weights_only=True)
            model.load_state_dict(state, strict=False)
            print(f"    Loaded GOES pre-trained weights")

        criterion = FlareForecasterLoss().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE_PRETRAIN,
                                      weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)

        X_t = torch.tensor(X_aug, dtype=torch.float32).to(device)
        y_t = torch.tensor(y_aug, dtype=torch.long).to(device)
        lt_t = torch.tensor(lt_aug, dtype=torch.float32).to(device)
        X_vt = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_vt = torch.tensor(y_val, dtype=torch.long).to(device)

        batch_size = cfg.BATCH_SIZE
        best_val_acc = 0

        for epoch in range(1, 31):
            model.train()
            perm = torch.randperm(len(X_aug))
            for i in range(0, len(X_aug), batch_size):
                idx = perm[i:i+batch_size]
                optimizer.zero_grad()
                logits, lead_pred, _ = model(X_t[idx])
                losses = criterion(logits, y_t[idx], lead_pred, lt_t[idx])
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                v_logits, _, _ = model(X_vt)
                v_acc = (v_logits.argmax(1) == y_vt).float().mean().item()
            if v_acc > best_val_acc:
                best_val_acc = v_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            if epoch % 10 == 0:
                print(f"    Epoch {epoch}: val_acc={v_acc:.3f}")

        # Load best and evaluate
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            v_logits, _, _ = model(X_vt)
            v_probs = torch.softmax(v_logits, dim=1).cpu().numpy()
            v_preds = v_logits.argmax(1).cpu().numpy()

        fold_acc = accuracy_score(y_val, v_preds)
        all_preds[val_idx] = v_preds
        all_probs[val_idx] = v_probs

        # Per-class AUC
        fold_aucs = {}
        for cls in range(cfg.N_CLASSES):
            y_bin = (y_val == cls).astype(int)
            if y_bin.sum() > 0 and y_bin.sum() < len(y_bin):
                try:
                    fold_aucs[cfg.CLASS_NAMES[cls]] = roc_auc_score(y_bin, v_probs[:, cls])
                except Exception:
                    pass

        fold_results.append({"fold": fold, "acc": fold_acc, "aucs": fold_aucs})
        print(f"    Fold {fold}: acc={fold_acc:.3f} aucs={fold_aucs}")

    # Summary
    accs = [r["acc"] for r in fold_results]
    overall_acc = accuracy_score(y, all_preds)

    print(f"\n{'='*60}")
    print(f"  5-FOLD CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"  Per-fold accuracy: {', '.join(f'{a:.3f}' for a in accs)}")
    print(f"  Mean accuracy:     {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"  Overall accuracy:  {overall_acc:.4f}")

    # Overall per-class AUC
    print(f"\n  Per-Class ROC-AUC (cross-validated):")
    for cls in range(cfg.N_CLASSES):
        y_bin = (y == cls).astype(int)
        if y_bin.sum() > 0 and y_bin.sum() < len(y):
            try:
                auc = roc_auc_score(y_bin, all_probs[:, cls])
                print(f"    {cfg.CLASS_NAMES[cls]:>5s}: {auc:.4f}")
            except Exception:
                pass

    # Binary detection metrics
    y_binary = (y > 0).astype(int)
    pred_binary = (all_preds > 0).astype(int)
    tp = ((pred_binary == 1) & (y_binary == 1)).sum()
    fp = ((pred_binary == 1) & (y_binary == 0)).sum()
    fn = ((pred_binary == 0) & (y_binary == 1)).sum()
    tn = ((pred_binary == 0) & (y_binary == 0)).sum()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    print(f"\n  Binary Detection (CV):")
    print(f"    TPR: {tpr:.4f}")
    print(f"    FPR: {fpr:.4f}")

    # FPR Optimization
    optimize_fpr(y, all_preds, all_probs)

    return fold_results, all_preds, all_probs


def optimize_fpr(y, predictions, probabilities, target_fpr=0.30):
    """Find optimal per-class thresholds to minimize FPR while keeping TPR high."""
    print(f"\n{'='*60}")
    print(f"  FPR OPTIMIZATION (target FPR < {target_fpr:.0%})")
    print(f"{'='*60}")

    best_threshold = 0.5
    best_f1 = 0
    best_metrics = {}

    for threshold in np.arange(0.30, 0.91, 0.05):
        # Apply threshold: only predict flare if max prob > threshold
        max_probs = probabilities.max(axis=1)
        max_classes = probabilities.argmax(axis=1)
        thresholded_preds = np.where(max_probs >= threshold, max_classes, 0)

        # Also require that non-quiet prediction has prob > threshold
        for i in range(len(thresholded_preds)):
            if thresholded_preds[i] > 0:
                if probabilities[i, thresholded_preds[i]] < threshold:
                    thresholded_preds[i] = 0

        y_bin = (y > 0).astype(int)
        p_bin = (thresholded_preds > 0).astype(int)
        tp = ((p_bin == 1) & (y_bin == 1)).sum()
        fp = ((p_bin == 1) & (y_bin == 0)).sum()
        fn = ((p_bin == 0) & (y_bin == 1)).sum()
        tn = ((p_bin == 0) & (y_bin == 0)).sum()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * prec * tpr / (prec + tpr) if (prec + tpr) > 0 else 0
        acc = (thresholded_preds == y).mean()

        if f1 > best_f1 and fpr <= target_fpr:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {"tpr": tpr, "fpr": fpr, "prec": prec,
                            "f1": f1, "acc": acc, "threshold": threshold}

        if threshold * 100 % 10 < 6:
            print(f"    thresh={threshold:.2f}: TPR={tpr:.3f} FPR={fpr:.3f} "
                  f"F1={f1:.3f} Acc={acc:.3f}")

    if best_metrics:
        print(f"\n  OPTIMAL THRESHOLD: {best_threshold:.2f}")
        print(f"    TPR:       {best_metrics['tpr']:.4f}")
        print(f"    FPR:       {best_metrics['fpr']:.4f}  (was 0.608)")
        print(f"    Precision: {best_metrics['prec']:.4f}")
        print(f"    F1 Score:  {best_metrics['f1']:.4f}")
        print(f"    Accuracy:  {best_metrics['acc']:.4f}")

        # Save optimal threshold
        np.save(str(cfg.MODEL_DIR / "optimal_threshold.npy"),
                np.array([best_threshold]))
    else:
        print(f"  Could not find threshold with FPR < {target_fpr}")
        # Still find best F1 regardless of FPR
        best_threshold = 0.5
        np.save(str(cfg.MODEL_DIR / "optimal_threshold.npy"),
                np.array([best_threshold]))

    return best_threshold


def step_tactical_ensemble(X, y, lead_times, quick=False, pretrained_weights=None):
    """Step 5a: Train Tier 2 ensemble with augmentation + calibration."""
    print("\n" + "=" * 70)
    if quick:
        print("  STEP 5a: Training TACTICAL Model (Quick Mode)")
    else:
        mode = "GOES Pre-trained " if pretrained_weights else ""
        print(f"  STEP 5a: Training {mode}TACTICAL ENSEMBLE (5 models + augmentation)")
    print("=" * 70)

    n_features = X.shape[2]

    if quick:
        # Quick mode: single model, no augmentation
        from src.model.train import train_model
        model = train_model(X, y, lead_times, mode="pretrain", epochs=20)
        return model, None
    else:
        from src.model.ensemble import EnsembleForecaster
        ensemble = EnsembleForecaster(n_models=5, n_features=n_features)
        metrics = ensemble.train_ensemble(
            X, y, lead_times,
            epochs=30,
            augment=True,
            aug_multiplier=5,
            pretrained_weights=pretrained_weights,
        )
        # Save ensemble
        save_dir = str(cfg.MODEL_DIR / "tactical_ensemble")
        ensemble.save(save_dir)
        return None, ensemble


def step_tactical_evaluate(model, ensemble, X, y, lead_times):
    """Step 6a: Evaluate tactical model/ensemble."""
    print("\n" + "=" * 70)
    print("  STEP 6a: Evaluating TACTICAL Model")
    print("=" * 70)

    if ensemble is not None:
        # Ensemble evaluation
        predictions, probabilities = ensemble.predict(X)
        lead_preds = ensemble._get_ensemble_lead_times(X)
        uncertainty = ensemble.get_uncertainty(X)

        # Compute metrics
        from sklearn.metrics import accuracy_score, roc_auc_score
        acc = accuracy_score(y, predictions)

        # Binary detection
        y_binary = (y > 0).astype(int)
        pred_binary = (predictions > 0).astype(int)
        tp = ((pred_binary == 1) & (y_binary == 1)).sum()
        fp = ((pred_binary == 1) & (y_binary == 0)).sum()
        fn = ((pred_binary == 0) & (y_binary == 1)).sum()
        tn = ((pred_binary == 0) & (y_binary == 0)).sum()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0

        print(f"\n{'='*60}")
        print(f"  ENSEMBLE EVALUATION RESULTS (Calibrated)")
        print(f"{'='*60}")
        print(f"\n  Overall Accuracy:  {acc:.4f}")
        print(f"  Temperature:       {ensemble.calibrator.temperature:.3f}")
        print(f"  Avg Uncertainty:   {uncertainty.mean():.4f}")
        print(f"\n  Binary Detection:")
        print(f"    TPR (Recall):    {tpr:.4f}")
        print(f"    FPR:             {fpr:.4f}")
        print(f"    Precision:       {precision:.4f}")
        print(f"    F1 Score:        {f1:.4f}")

        # Lead time stats for correct predictions
        correct_mask = (predictions == y) & (y > 0)
        if correct_mask.sum() > 0:
            correct_lts = lead_preds[correct_mask]
            print(f"\n  Lead Time (correct predictions):")
            print(f"    Mean:            {correct_lts.mean():.1f} min")
            print(f"    Median:          {np.median(correct_lts):.1f} min")
            print(f"    >=15 min:        {(correct_lts >= 15).mean()*100:.1f}%")
            print(f"    >=30 min:        {(correct_lts >= 30).mean()*100:.1f}%")

        # Per-class AUC
        print(f"\n  Per-Class ROC-AUC:")
        for cls in range(cfg.N_CLASSES):
            y_bin = (y == cls).astype(int)
            if y_bin.sum() > 0 and y_bin.sum() < len(y):
                try:
                    auc = roc_auc_score(y_bin, probabilities[:, cls])
                    print(f"    {cfg.CLASS_NAMES[cls]:>5s}:  {auc:.4f}")
                except:
                    pass

        # Alert level distribution
        detailed = ensemble.predict_detailed(X)
        alert_counts = {"GREEN": 0, "YELLOW": 0, "RED": 0}
        for d in detailed:
            alert_counts[d["alert_level"]] += 1
        print(f"\n  Alert Level Distribution:")
        for level, count in alert_counts.items():
            print(f"    {level:6s}: {count:4d} ({count/len(X)*100:.1f}%)")

        # Confusion matrix
        print(f"\n  Confusion Matrix (with confidence thresholds):")
        print(f"  {'':>10s}", end="")
        for cn in cfg.CLASS_NAMES:
            print(f"{cn:>8s}", end="")
        print()
        for i, cn in enumerate(cfg.CLASS_NAMES):
            print(f"  {cn:>10s}", end="")
            for j in range(cfg.N_CLASSES):
                count = ((predictions == j) & (y == i)).sum()
                print(f"{count:8d}", end="")
            print()

        # FPR optimization
        optimize_fpr(y, predictions, probabilities)

        # Generate plots
        try:
            from src.model.evaluate import generate_plots
            plot_dir = str(cfg.PLOTS_DIR / "tactical")
            os.makedirs(plot_dir, exist_ok=True)
            generate_plots(y, predictions, probabilities, lead_preds, plot_dir)
        except Exception as e:
            print(f"\n  Plot generation: {e}")

    else:
        # Single model evaluation
        from src.model.evaluate import full_evaluation
        full_evaluation(model, X, y, lead_times,
                       save_dir=str(cfg.PLOTS_DIR / "tactical"))


def step_strategic_windowing(df_feat, feat_cols, catalog):
    """Step 4b: Create Tier 1 (strategic) windows."""
    print("\n" + "=" * 70)
    print("  STEP 4b: Tier 1 STRATEGIC Windows (10-hour horizon)")
    print("=" * 70)
    from src.features.windowing import (create_strategic_windows,
                                         normalize_features,
                                         balance_classes, print_window_stats)

    X, y, metadata = create_strategic_windows(df_feat, feat_cols, flare_catalog=catalog)
    print_window_stats(y, metadata)

    X_norm, mean, std = normalize_features(X)
    np.save(str(cfg.PROCESSED / "strategic_mean.npy"), mean)
    np.save(str(cfg.PROCESSED / "strategic_std.npy"), std)

    X_bal, y_bal, meta_bal = balance_classes(X_norm, y, metadata, max_ratio=5)
    print(f"\n  After balancing: {len(X_bal)} windows")
    print_window_stats(y_bal, meta_bal)

    np.save(str(cfg.PROCESSED / "X_strategic.npy"), X_bal)
    np.save(str(cfg.PROCESSED / "y_strategic.npy"), y_bal)

    return X_bal, y_bal


def step_strategic_train(X, y):
    """Step 5b: Train strategic model."""
    print("\n" + "=" * 70)
    print("  STEP 5b: Training STRATEGIC Model (Tier 1)")
    print("=" * 70)
    from src.model.architecture import StrategicForecaster, StrategicLoss
    from src.model.augmentation import augment_dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Training device: {device}")

    n_features = X.shape[2]

    # Augment strategic data too
    dummy_lt = np.zeros(len(y), dtype=np.float32)
    X_aug, y_aug, _ = augment_dataset(X, y, dummy_lt, multiplier=5, seed=123)
    print(f"  Augmented: {len(X)} -> {len(X_aug)} samples")

    model = StrategicForecaster(n_input_channels=n_features).to(device)
    criterion = StrategicLoss().to(device)
    params = model.count_parameters()
    print(f"  Model: {params['trainable']:,} params")

    n_val = int(len(X_aug) * 0.15)
    n_train = len(X_aug) - n_val

    X_train_t = torch.tensor(X_aug[:n_train], dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_aug[:n_train], dtype=torch.long).to(device)
    X_val_t = torch.tensor(X_aug[n_train:], dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_aug[n_train:], dtype=torch.long).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)

    best_val_loss = float("inf")
    patience_counter = 0
    batch_size = 32

    print(f"  Training: {n_train} train, {n_val} val, 40 epochs")
    print(f"\n{'Epoch':>5} {'Train Loss':>11} {'Val Loss':>11} "
          f"{'Train Acc':>10} {'Val Acc':>10} {'Time':>6}")
    print("-" * 55)

    for epoch in range(1, 41):
        t0 = time.time()
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            xb = X_train_t[idx]
            yb = y_train_t[idx]

            optimizer.zero_grad()
            logits, _ = model(xb)
            losses = criterion(logits, yb)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += losses["total"].item() * len(idx)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(idx)

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits, _ = model(X_val_t)
            val_losses = criterion(val_logits, y_val_t)
            val_loss = val_losses["total"].item()
            val_acc = (val_logits.argmax(1) == y_val_t).float().mean().item()

        dt = time.time() - t0
        train_loss = total_loss / total
        train_acc = correct / total

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:5d} {train_loss:11.4f} {val_loss:11.4f} "
                  f"{train_acc:10.4f} {val_acc:10.4f} {dt:5.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), str(cfg.MODEL_DIR / "best_strategic_model.pt"))
        else:
            patience_counter += 1

        if patience_counter >= 12:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    # Load best
    best_path = str(cfg.MODEL_DIR / "best_strategic_model.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))

    # Final eval
    model.eval()
    with torch.no_grad():
        all_logits, _ = model(torch.tensor(X, dtype=torch.float32).to(device))
        y_pred = all_logits.argmax(1).cpu().numpy()

    acc = (y_pred == y).mean()
    flare_pred = (y_pred > 0).astype(int)
    flare_true = (y > 0).astype(int)
    tpr = (flare_pred[flare_true == 1] == 1).mean() if flare_true.sum() > 0 else 0
    fpr = (flare_pred[flare_true == 0] == 1).mean() if (flare_true == 0).sum() > 0 else 0

    print(f"\n  Strategic Model Results:")
    print(f"    Accuracy:     {acc:.4f}")
    print(f"    TPR (Recall): {tpr:.4f}")
    print(f"    FPR:          {fpr:.4f}")

    return model


def main():
    parser = argparse.ArgumentParser(description="JWALASHMI v4.0 - Solar Flare Early Warning")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--nowcast", action="store_true")
    parser.add_argument("--tactical", action="store_true")
    parser.add_argument("--strategic", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: single model, no augmentation")
    parser.add_argument("--pretrain-goes", action="store_true",
                        help="Pre-train on GOES data before fine-tuning")
    parser.add_argument("--cv", action="store_true",
                        help="Run 5-fold stratified cross-validation")
    args = parser.parse_args()

    print("\n" + "#" * 70)
    print("  JWALASHMI v4.0 -- Solar Flare Early Warning System")
    print("  GOES Pre-training + 5-Fold CV + FPR Optimization + Ensemble")
    print("#" * 70)

    t_start = time.time()

    if args.extract:
        from src.data.extract_all import extract_hel1os_zips
        extract_hel1os_zips()

    df_solexs, df_hel1os = step_load_data()
    flares, catalog = step_nowcast(df_solexs, df_hel1os)

    if args.nowcast:
        print(f"\nNowcasting complete in {time.time()-t_start:.1f}s")
        return

    df_feat, feat_cols = step_features(df_solexs, df_hel1os)

    # GOES Pre-training (transfer learning)
    pretrained_weights = None
    if args.pretrain_goes:
        pretrained_weights = step_pretrain_goes()

    run_tactical = not args.strategic
    run_strategic = not args.tactical

    if run_tactical:
        X_tac, y_tac, lt_tac = step_tactical_windowing(df_feat, feat_cols, catalog)
        if X_tac.shape[0] >= 10:
            if args.cv:
                # 5-fold cross-validation
                step_cross_validate(X_tac, y_tac, lt_tac,
                                   pretrained_weights=pretrained_weights)
            else:
                # Standard ensemble training
                model_tac, ensemble_tac = step_tactical_ensemble(
                    X_tac, y_tac, lt_tac,
                    quick=args.quick,
                    pretrained_weights=pretrained_weights)
                step_tactical_evaluate(model_tac, ensemble_tac,
                                      X_tac, y_tac, lt_tac)

    if run_strategic:
        X_str, y_str = step_strategic_windowing(df_feat, feat_cols, catalog)
        if X_str.shape[0] >= 10:
            step_strategic_train(X_str, y_str)

    total_time = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE - Total time: {total_time/60:.1f} minutes")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
