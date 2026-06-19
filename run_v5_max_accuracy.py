"""
JWALASHMI v5.1 - Maximum Accuracy with Fresh Samples Every Epoch
Each epoch generates BRAND NEW augmented samples - model never sees same data twice.

Key innovation: Online augmentation per epoch instead of static pre-augmentation.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import config as cfg


def online_augment_batch(X_batch, rng):
    """Fully vectorized augmentation - NO per-sample loops."""
    X_aug = X_batch.copy()
    N, T, C = X_aug.shape

    # Split batch into groups, each gets a different augmentation
    indices = np.arange(N)
    rng.shuffle(indices)
    chunk = max(N // 5, 1)

    # Group 1: Gaussian noise
    g1 = indices[:chunk]
    if len(g1) > 0:
        stds = np.std(X_aug[g1], axis=1, keepdims=True).clip(1e-8)
        X_aug[g1] += rng.normal(0, 1, X_aug[g1].shape) * stds * 0.05

    # Group 2: Amplitude scale
    g2 = indices[chunk:2*chunk]
    if len(g2) > 0:
        scales = rng.uniform(0.75, 1.25, size=(len(g2), 1, 1))
        X_aug[g2] *= scales

    # Group 3: Time shift (vectorized roll via indexing)
    g3 = indices[2*chunk:3*chunk]
    if len(g3) > 0:
        for i in g3:
            shift = rng.integers(-300, 301)
            X_aug[i] = np.roll(X_aug[i], shift, axis=0)

    # Group 4: Feature dropout
    g4 = indices[3*chunk:4*chunk]
    if len(g4) > 0:
        drop_mask = rng.random((len(g4), 1, C)) < 0.15
        X_aug[g4] *= (~drop_mask).astype(np.float32)

    # Group 5: Combined noise + scale
    g5 = indices[4*chunk:]
    if len(g5) > 0:
        stds = np.std(X_aug[g5], axis=1, keepdims=True).clip(1e-8)
        X_aug[g5] += rng.normal(0, 1, X_aug[g5].shape) * stds * 0.04
        scales = rng.uniform(0.85, 1.15, size=(len(g5), 1, 1))
        X_aug[g5] *= scales

    return X_aug.astype(np.float32)


def mixup_batch(X1, y1, X2, y2, alpha=0.3, rng=None):
    """Mixup two batches."""
    rng = rng or np.random.default_rng()
    lam = rng.beta(alpha, alpha)
    X_mix = (lam * X1 + (1 - lam) * X2).astype(np.float32)
    y_mix = np.where(rng.random(len(y1)) < lam, y1, y2)
    return X_mix, y_mix


def oversample_minorities(X, y, lead_times, target_per_class=200):
    """SMOTE-like oversampling: repeat + noise for minority classes."""
    rng = np.random.default_rng(99)
    unique_classes = np.unique(y)
    all_X, all_y, all_lt = [X.copy()], [y.copy()], [lead_times.copy()]

    for cls in unique_classes:
        mask = y == cls
        count = mask.sum()
        if count < target_per_class:
            needed = target_per_class - count
            X_cls = X[mask]
            y_cls = y[mask]
            lt_cls = lead_times[mask]

            # Repeat with noise
            repeats = int(np.ceil(needed / count))
            for r in range(repeats):
                X_new = X_cls.copy()
                # Add unique noise each repeat
                for i in range(len(X_new)):
                    for f in range(X_new.shape[2]):
                        std = np.std(X_new[i, :, f])
                        if std > 0:
                            X_new[i, :, f] += rng.normal(0, 0.05 * std, X_new.shape[1])
                    # Small random scale
                    X_new[i] *= rng.uniform(0.9, 1.1)

                all_X.append(X_new[:min(needed, len(X_new))])
                all_y.append(y_cls[:min(needed, len(y_cls))])
                all_lt.append(lt_cls[:min(needed, len(lt_cls))])
                needed -= len(X_new)
                if needed <= 0:
                    break

    X_out = np.concatenate(all_X, axis=0)
    y_out = np.concatenate(all_y, axis=0)
    lt_out = np.concatenate(all_lt, axis=0)

    perm = rng.permutation(len(X_out))
    return X_out[perm], y_out[perm], lt_out[perm]


def train_fresh_samples_ensemble(X, y, lead_times, pretrained_weights=None):
    """
    10-model ensemble where EVERY EPOCH generates fresh augmented samples.
    No model ever sees the same augmented data twice.
    """
    print("\n" + "=" * 70)
    print("  v5.3 FRESH-SAMPLE ENSEMBLE (10 models x 50 epochs)")
    print("  Every epoch = brand new augmented samples")
    print("  GPU cooling pauses + mixed precision enabled")
    print("=" * 70)

    from src.model.architecture import FlareForecaster, FlareForecasterLoss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    print(f"  Device: {device}, Mixed Precision: {use_amp}")
    n_features = X.shape[2]
    n_models = 10
    models = []

    # First oversample minorities so each class has ~200 samples
    X_os, y_os, lt_os = oversample_minorities(X, y, lead_times, target_per_class=200)
    print(f"\n  After SMOTE oversampling: {len(X_os)} samples")
    for c in range(cfg.N_CLASSES):
        print(f"    {cfg.CLASS_NAMES[c]}: {(y_os == c).sum()}")

    for model_idx in range(n_models):
        # Cooling pause between models (prevent laptop overheating)
        if model_idx > 0 and device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            print(f"\n  [COOLING] 30s pause to prevent overheating...")
            time.sleep(30)

        seed = model_idx * 53 + 13
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"\n--- Model {model_idx+1}/{n_models} (seed={seed}) ---")

        # Split: 85% train, 15% val (from oversampled data)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(X_os))
        n_val = max(int(len(X_os) * 0.15), 10)
        n_train = len(X_os) - n_val

        X_train_raw = X_os[perm[:n_train]]
        y_train_raw = y_os[perm[:n_train]]
        lt_train_raw = lt_os[perm[:n_train]]
        X_val = torch.tensor(X_os[perm[n_train:]], dtype=torch.float32).to(device)
        y_val = torch.tensor(y_os[perm[n_train:]], dtype=torch.long).to(device)

        print(f"  Train: {n_train}, Val: {n_val}")

        # Create model
        model = FlareForecaster(n_input_channels=n_features).to(device)

        # Load pre-trained weights
        if pretrained_weights and os.path.exists(pretrained_weights):
            state = torch.load(pretrained_weights, map_location=device, weights_only=True)
            model.load_state_dict(state, strict=False)
            if model_idx == 0:
                print(f"  Loaded GOES pre-trained weights")

        lr = 3e-4 if pretrained_weights else 5e-4
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=15, T_mult=2, eta_min=1e-6
        )

        batch_size = 32
        best_val_acc = 0
        best_state = None
        epochs = 50
        label_smooth = 0.1
        epoch_rng_base = seed * 10000

        for epoch in range(1, epochs + 1):
            model.train()

            # === FRESH SAMPLES EVERY EPOCH ===
            epoch_rng = np.random.default_rng(epoch_rng_base + epoch)

            # Generate fresh augmented data for this epoch
            X_aug = online_augment_batch(X_train_raw, epoch_rng)

            # Also create a mixup batch (fresh pairs)
            perm2 = epoch_rng.permutation(n_train)
            X_mix, y_mix = mixup_batch(
                X_train_raw, y_train_raw,
                X_train_raw[perm2], y_train_raw[perm2],
                alpha=0.3, rng=epoch_rng
            )

            # Combine: original + augmented + mixup
            X_epoch = np.concatenate([X_train_raw, X_aug, X_mix], axis=0)
            y_epoch = np.concatenate([y_train_raw, y_train_raw, y_mix], axis=0)
            lt_epoch = np.concatenate([lt_train_raw, lt_train_raw,
                                       np.zeros(len(y_mix), dtype=np.float32)], axis=0)

            # Shuffle this epoch's data
            epoch_perm = epoch_rng.permutation(len(X_epoch))
            X_epoch = X_epoch[epoch_perm]
            y_epoch = y_epoch[epoch_perm]
            lt_epoch = lt_epoch[epoch_perm]

            X_t = torch.tensor(X_epoch, dtype=torch.float32).to(device)
            y_t = torch.tensor(y_epoch, dtype=torch.long).to(device)
            lt_t = torch.tensor(lt_epoch, dtype=torch.float32).to(device)

            total_loss = 0
            correct = 0
            total = 0

            for i in range(0, len(X_epoch), batch_size):
                end = min(i + batch_size, len(X_epoch))
                if end - i < 2:
                    continue

                optimizer.zero_grad()

                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits, lead_pred, _ = model(X_t[i:end])

                    # Label smoothing
                    n_cls = logits.shape[1]
                    smooth_target = torch.full_like(logits, label_smooth / (n_cls - 1))
                    smooth_target.scatter_(1, y_t[i:end].unsqueeze(1), 1.0 - label_smooth)
                    log_probs = F.log_softmax(logits, dim=1)
                    loss_cls = -(smooth_target * log_probs).sum(dim=1).mean()

                    # Lead time loss
                    flare_mask = y_t[i:end] > 0
                    if flare_mask.any():
                        lt_pred = lead_pred[flare_mask].squeeze(-1)
                        lt_true = lt_t[i:end][flare_mask]
                        loss_lt = F.mse_loss(lt_pred, lt_true) * 0.1
                    else:
                        loss_lt = torch.tensor(0.0, device=device)

                    loss = loss_cls + loss_lt

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                total_loss += loss.item() * (end - i)
                correct += (logits.argmax(1) == y_t[i:end]).sum().item()
                total += (end - i)

            scheduler.step()

            # Validation
            model.eval()
            with torch.no_grad():
                v_logits, _, _ = model(X_val)
                v_acc = (v_logits.argmax(1) == y_val).float().mean().item()

            if v_acc > best_val_acc:
                best_val_acc = v_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            if epoch % 10 == 0 or epoch == 1:
                train_acc = correct / max(total, 1)
                print(f"  Epoch {epoch:3d}: loss={total_loss/max(total,1):.3f} "
                      f"train={train_acc:.3f} val={v_acc:.3f} "
                      f"(fresh {len(X_epoch)} samples)")

        if best_state:
            model.load_state_dict(best_state)
        model.eval()
        models.append(model)
        print(f"  Best val_acc: {best_val_acc:.3f}")

    # === ENSEMBLE EVALUATION ===
    print("\n" + "=" * 70)
    print("  ENSEMBLE EVALUATION (10 models)")
    print("=" * 70)

    # Evaluate on the ORIGINAL balanced data
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    all_probs = []
    for model in models:
        model.eval()
        with torch.no_grad():
            logits, _, _ = model(X_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)

    ensemble_probs = np.mean(all_probs, axis=0)
    ensemble_preds = ensemble_probs.argmax(axis=1)

    from sklearn.metrics import accuracy_score, roc_auc_score

    acc = accuracy_score(y, ensemble_preds)
    print(f"\n  5-Class Accuracy: {acc*100:.1f}%")

    # Per-class AUC
    print(f"\n  Per-Class ROC-AUC:")
    for cls in range(cfg.N_CLASSES):
        y_bin = (y == cls).astype(int)
        if 0 < y_bin.sum() < len(y):
            try:
                auc = roc_auc_score(y_bin, ensemble_probs[:, cls])
                print(f"    {cfg.CLASS_NAMES[cls]:>5s}: {auc:.4f}")
            except Exception:
                pass

    # Binary detection
    y_bin = (y > 0).astype(int)
    p_bin = (ensemble_preds > 0).astype(int)
    tp = ((p_bin == 1) & (y_bin == 1)).sum()
    fp = ((p_bin == 1) & (y_bin == 0)).sum()
    fn = ((p_bin == 0) & (y_bin == 1)).sum()
    tn = ((p_bin == 0) & (y_bin == 0)).sum()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    print(f"\n  Binary Detection: TPR={tpr:.3f} FPR={fpr:.3f}")

    # 3-Tier Alert
    tier_map = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}
    y_tier = np.array([tier_map[c] for c in y])
    p_tier = np.array([tier_map[c] for c in ensemble_preds])
    tier_acc = accuracy_score(y_tier, p_tier)
    tier_names = ["GREEN", "YELLOW", "RED"]
    print(f"\n  3-Tier Alert Accuracy: {tier_acc*100:.1f}%")
    for t in range(3):
        mask = y_tier == t
        if mask.sum() > 0:
            t_acc = (p_tier[mask] == t).mean()
            print(f"    {tier_names[t]:>6s}: {t_acc*100:.1f}% ({mask.sum()} samples)")

    # Confusion matrix
    print(f"\n  Confusion Matrix:")
    print(f"  {'':>10s}", end="")
    for cn in cfg.CLASS_NAMES:
        print(f"{cn:>8s}", end="")
    print()
    for i, cn in enumerate(cfg.CLASS_NAMES):
        print(f"  {cn:>10s}", end="")
        for j in range(cfg.N_CLASSES):
            count = ((ensemble_preds == j) & (y == i)).sum()
            print(f"{count:8d}", end="")
        print()

    # Threshold sweep
    print(f"\n  Threshold Sweep:")
    for threshold in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        max_p = ensemble_probs.max(axis=1)
        max_c = ensemble_probs.argmax(axis=1)
        thr_preds = np.where(max_p >= threshold, max_c, 0)
        t_acc = accuracy_score(y, thr_preds)
        # 3-tier with threshold
        p_tier_t = np.array([tier_map[c] for c in thr_preds])
        t3_acc = accuracy_score(y_tier, p_tier_t)
        print(f"    thresh={threshold:.1f}: 5-class={t_acc*100:.1f}% 3-tier={t3_acc*100:.1f}%")

    # Save
    save_dir = str(cfg.MODEL_DIR / "v5_ensemble")
    os.makedirs(save_dir, exist_ok=True)
    for i, model in enumerate(models):
        torch.save(model.state_dict(), os.path.join(save_dir, f"model_{i}.pt"))
    print(f"\n  Saved ensemble to {save_dir}")

    return models, ensemble_probs, ensemble_preds


def main():
    import pandas as pd

    print("\n" + "#" * 70)
    print("  JWALASHMI v5.1 -- FRESH SAMPLES EVERY EPOCH")
    print("  10-Model Ensemble + Online Augmentation + SMOTE + Mixup")
    print("#" * 70)

    t_start = time.time()

    # Step 1: Load
    print("\n" + "=" * 70)
    print("  STEP 1: Loading Data")
    print("=" * 70)
    from src.data.fits_loader import (load_solexs_all, load_hel1os_all,
                                       find_solexs_files, find_hel1os_files)
    print(f"  SoLEXS: {len(find_solexs_files())} days, HEL1OS: {len(find_hel1os_files())} files")
    df_solexs = load_solexs_all()
    df_hel1os = load_hel1os_all(detector="cdte1")

    # Step 2: Nowcast
    print("\n" + "=" * 70)
    print("  STEP 2: Nowcasting")
    print("=" * 70)
    from src.nowcasting.detector import detect_flares, build_unified_catalog
    all_flares = []
    for date, group in df_solexs.groupby("date"):
        flares = detect_flares(group.reset_index(drop=True), instrument="solexs")
        all_flares.extend(flares)
        if flares:
            print(f"  {date}: {len(flares)} flares")
    if not df_hel1os.empty:
        broad_col = [c for c in df_hel1os.columns if "1.80KEV_TO_90" in c]
        if broad_col:
            for date, group in df_hel1os.groupby("date"):
                fl = detect_flares(group.reset_index(drop=True),
                                   instrument="hel1os", count_col=broad_col[0])
                all_flares.extend(fl)
    catalog = build_unified_catalog(all_flares, [])
    catalog.to_csv(str(cfg.CATALOG_CSV), index=False)
    print(f"  Catalog: {len(catalog)} events")
    dist = {}
    for c in catalog["estimated_class"]:
        dist[c] = dist.get(c, 0) + 1
    print(f"  Distribution: {dist}")

    # Step 3: Features (SoLEXS + HEL1OS merged)
    print("\n" + "=" * 70)
    print("  STEP 3: Features (SoLEXS + HEL1OS)")
    print("=" * 70)
    from src.features.physics_features import compute_all_features, get_feature_columns

    # Merge HEL1OS data with SoLEXS by timestamp
    hel1os_dates = set()
    if not df_hel1os.empty:
        hel1os_dates = set(df_hel1os["date"].unique())
        print(f"  HEL1OS dates available: {sorted(hel1os_dates)}")

    # Identify HEL1OS column names for physics features
    hel1os_ctr_cols = [c for c in df_hel1os.columns if c.startswith("ctr_")] if not df_hel1os.empty else []
    hxr_soft_col = next((c for c in hel1os_ctr_cols if "5.00KEV_TO_20" in c), None)
    hxr_hard_col = next((c for c in hel1os_ctr_cols if "30.00KEV_TO_40" in c), None)
    hxr_medium_col = next((c for c in hel1os_ctr_cols if "20.00KEV_TO_30" in c), None)
    hxr_broad_col = next((c for c in hel1os_ctr_cols if "1.80KEV_TO_90" in c), None)

    if hxr_soft_col:
        print(f"  HEL1OS bands: soft={hxr_soft_col}, hard={hxr_hard_col}, broad={hxr_broad_col}")
        hel1os_feat_cols = {
            "hxr_soft": hxr_soft_col,
            "hxr_hard": hxr_hard_col,
            "hxr_medium": hxr_medium_col,
        }
    else:
        hel1os_feat_cols = None
        print("  HEL1OS: No matching energy bands found")

    dfs = []
    for date in df_solexs["date"].unique():
        day_solexs = df_solexs[df_solexs["date"] == date].copy().reset_index(drop=True)

        if date in hel1os_dates and not df_hel1os.empty:
            # Merge HEL1OS with SoLEXS for this date
            day_hel1os = df_hel1os[df_hel1os["date"] == date].copy()
            day_solexs["ts_round"] = day_solexs["timestamp"].round(0)
            day_hel1os["ts_round"] = day_hel1os["timestamp"].round(0)

            merge_cols = ["ts_round"] + [c for c in day_hel1os.columns if c.startswith("ctr_")]
            day_merged = pd.merge(
                day_solexs, day_hel1os[merge_cols].drop_duplicates(subset="ts_round"),
                on="ts_round", how="left"
            ).drop(columns=["ts_round"])

            # Fill NaN HEL1OS values with 0
            for col in [c for c in day_merged.columns if c.startswith("ctr_")]:
                day_merged[col] = day_merged[col].fillna(0)

            day_feat = compute_all_features(day_merged, hel1os_cols=hel1os_feat_cols)
            print(f"  {date}: SoLEXS+HEL1OS merged ({len(day_merged)} rows)")
        else:
            day_feat = compute_all_features(day_solexs)
            # Add zero columns for HEL1OS features to keep feature count consistent
            for fcol in ["feat_hard_soft_ratio", "feat_neupert", "feat_spectral_hardness"]:
                if fcol not in day_feat.columns:
                    day_feat[fcol] = 0.0

        dfs.append(day_feat)

    df_feat = pd.concat(dfs, ignore_index=True)
    feat_cols = get_feature_columns(df_feat)
    print(f"  {len(feat_cols)} features: {feat_cols}")

    # Step 4: Windows
    print("\n" + "=" * 70)
    print("  STEP 4: Windowing")
    print("=" * 70)
    from src.features.windowing import (create_windows, normalize_features,
                                         balance_classes, print_window_stats)
    X, y, meta = create_windows(df_feat, feat_cols, flare_catalog=catalog)
    print_window_stats(y, meta)
    X_norm, mean, std = normalize_features(X)
    X_bal, y_bal, meta_bal = balance_classes(X_norm, y, meta, max_ratio=10)
    print(f"  After balancing: {len(X_bal)} windows")
    print_window_stats(y_bal, meta_bal)

    lead_times = np.array([
        m.lead_time / 60 if m.lead_time is not None else 0
        for m in meta_bal
    ], dtype=np.float32)

    # Step 5: Train
    pretrained_weights = str(cfg.MODEL_DIR / "goes_pretrained.pt")
    if not os.path.exists(pretrained_weights):
        pretrained_weights = None
        print("  No GOES pre-trained weights found")
    else:
        print(f"  Using GOES pre-trained: {pretrained_weights}")

    models, probs, preds = train_fresh_samples_ensemble(
        X_bal, y_bal, lead_times, pretrained_weights=pretrained_weights)

    total_time = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  v5.1 COMPLETE - Total time: {total_time/60:.1f} minutes")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
