"""
Solar Flare Early Warning System — Model Evaluation & Visualization
Generates all metrics ISRO evaluates: TPR, FPR, ROC-AUC, confusion matrix,
lead time distribution, and attention heatmaps.
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (confusion_matrix, classification_report,
                              roc_curve, auc, precision_recall_curve)
from typing import Dict, List, Optional, Tuple
import os

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config as cfg
from src.model.architecture import FlareForecaster


# ═══════════════════════════════════════════════════════════════
#  Metrics Computation
# ═══════════════════════════════════════════════════════════════

def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                        y_prob: np.ndarray,
                        lead_times_true: Optional[np.ndarray] = None,
                        lead_times_pred: Optional[np.ndarray] = None) -> Dict:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: (N,) true class labels
        y_pred: (N,) predicted class labels
        y_prob: (N, n_classes) predicted probabilities
        lead_times_true: (N,) true lead times in minutes
        lead_times_pred: (N,) predicted lead times in minutes
    
    Returns:
        Dictionary with all metrics
    """
    metrics = {}

    # Overall accuracy
    metrics["accuracy"] = (y_true == y_pred).mean()

    # Per-class metrics
    report = classification_report(y_true, y_pred,
                                    target_names=cfg.CLASS_NAMES,
                                    output_dict=True, zero_division=0)
    metrics["classification_report"] = report

    # Confusion matrix
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)

    # Binary detection metrics (flare vs no-flare)
    y_binary_true = (y_true > 0).astype(int)
    y_binary_pred = (y_pred > 0).astype(int)
    y_binary_prob = 1 - y_prob[:, 0]  # probability of ANY flare

    tn, fp, fn, tp = confusion_matrix(y_binary_true, y_binary_pred,
                                       labels=[0, 1]).ravel()
    metrics["TPR"] = tp / max(tp + fn, 1)  # True Positive Rate (Recall)
    metrics["FPR"] = fp / max(fp + tn, 1)  # False Positive Rate
    metrics["precision"] = tp / max(tp + fp, 1)
    metrics["F1"] = 2 * tp / max(2 * tp + fp + fn, 1)

    # ROC-AUC for binary detection
    if len(np.unique(y_binary_true)) > 1:
        fpr_curve, tpr_curve, _ = roc_curve(y_binary_true, y_binary_prob)
        metrics["ROC_AUC"] = auc(fpr_curve, tpr_curve)
        metrics["roc_curve"] = (fpr_curve, tpr_curve)
    else:
        metrics["ROC_AUC"] = 0.0

    # Per-class ROC-AUC (one-vs-rest)
    metrics["per_class_auc"] = {}
    for cls in range(cfg.N_CLASSES):
        cls_true = (y_true == cls).astype(int)
        if len(np.unique(cls_true)) > 1 and y_prob.shape[1] > cls:
            fpr_c, tpr_c, _ = roc_curve(cls_true, y_prob[:, cls])
            metrics["per_class_auc"][cfg.CLASS_NAMES[cls]] = auc(fpr_c, tpr_c)

    # Lead time metrics
    if lead_times_true is not None and lead_times_pred is not None:
        flare_mask = y_true > 0
        correct_mask = (y_true == y_pred) & flare_mask

        if correct_mask.sum() > 0:
            correct_leads = lead_times_true[correct_mask]
            metrics["mean_lead_time_min"] = float(np.mean(correct_leads))
            metrics["median_lead_time_min"] = float(np.median(correct_leads))
            metrics["lead_time_std_min"] = float(np.std(correct_leads))
            metrics["pct_above_15min"] = float((correct_leads >= 15).mean() * 100)
            metrics["pct_above_30min"] = float((correct_leads >= 30).mean() * 100)
        else:
            metrics["mean_lead_time_min"] = 0.0

    return metrics


def print_metrics(metrics: Dict):
    """Pretty-print evaluation metrics."""
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)

    print(f"\n  Overall Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Binary Detection:")
    print(f"    TPR (Recall):    {metrics['TPR']:.4f}")
    print(f"    FPR:             {metrics['FPR']:.4f}")
    print(f"    Precision:       {metrics['precision']:.4f}")
    print(f"    F1 Score:        {metrics['F1']:.4f}")
    print(f"    ROC-AUC:         {metrics['ROC_AUC']:.4f}")

    if metrics.get("mean_lead_time_min"):
        print(f"\n  Lead Time (correct predictions):")
        print(f"    Mean:            {metrics['mean_lead_time_min']:.1f} min")
        print(f"    Median:          {metrics.get('median_lead_time_min', 0):.1f} min")
        print(f"    >=15 min:        {metrics.get('pct_above_15min', 0):.1f}%")
        print(f"    >=30 min:        {metrics.get('pct_above_30min', 0):.1f}%")

    if metrics.get("per_class_auc"):
        print(f"\n  Per-Class ROC-AUC:")
        for cls, auc_val in metrics["per_class_auc"].items():
            print(f"    {cls:>5s}:  {auc_val:.4f}")

    print("\n  Confusion Matrix:")
    cm = metrics["confusion_matrix"]
    header = "         " + "  ".join(f"{n:>6s}" for n in cfg.CLASS_NAMES)
    print(f"  {header}")
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:6d}" for v in row)
        print(f"  {cfg.CLASS_NAMES[i]:>7s}  {row_str}")


# ═══════════════════════════════════════════════════════════════
#  Visualization
# ═══════════════════════════════════════════════════════════════

# Custom ISRO-themed colormap
ISRO_COLORS = ["#0a0a2e", "#1a1a5e", "#e65100", "#ff9800", "#ffd54f"]
ISRO_CMAP = LinearSegmentedColormap.from_list("isro", ISRO_COLORS, N=256)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                           save_path: Optional[str] = None) -> plt.Figure:
    """Plot confusion matrix with ISRO-themed styling."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_norm, cmap=ISRO_CMAP, vmin=0, vmax=1)

    ax.set_xticks(range(cfg.N_CLASSES))
    ax.set_yticks(range(cfg.N_CLASSES))
    ax.set_xticklabels(cfg.CLASS_NAMES, fontsize=12)
    ax.set_yticklabels(cfg.CLASS_NAMES, fontsize=12)
    ax.set_xlabel("Predicted", fontsize=14)
    ax.set_ylabel("True", fontsize=14)
    ax.set_title("Flare Classification - Confusion Matrix", fontsize=16, fontweight="bold")

    # Annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.0%})",
                    ha="center", va="center", color=color, fontsize=11)

    fig.colorbar(im, ax=ax, label="Recall", shrink=0.8)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_roc_curves(y_true: np.ndarray, y_prob: np.ndarray,
                     save_path: Optional[str] = None) -> plt.Figure:
    """Plot ROC curves for binary and per-class detection."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Binary ROC (flare vs no-flare)
    ax = axes[0]
    y_binary = (y_true > 0).astype(int)
    prob_flare = 1 - y_prob[:, 0]
    if len(np.unique(y_binary)) > 1:
        fpr, tpr, _ = roc_curve(y_binary, prob_flare)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color="#e65100", linewidth=2.5,
                label=f"Flare Detection (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Binary Flare Detection ROC", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Per-class ROC
    ax = axes[1]
    colors = ["#4caf50", "#2196f3", "#ff9800", "#f44336"]
    for cls in range(1, cfg.N_CLASSES):
        cls_true = (y_true == cls).astype(int)
        if len(np.unique(cls_true)) > 1:
            fpr, tpr, _ = roc_curve(cls_true, y_prob[:, cls])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[cls-1], linewidth=2,
                    label=f"{cfg.CLASS_NAMES[cls]} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Per-Class ROC Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Solar Flare Forecaster - ROC Analysis",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_attention_heatmap(model: FlareForecaster, X_sample: np.ndarray,
                            timestamps: Optional[np.ndarray] = None,
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize attention weights overlaid on the input time series.
    Shows WHERE the model is looking to make its prediction.
    """
    device = next(model.parameters()).device
    x_tensor = torch.tensor(X_sample, dtype=torch.float32).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits, lead_time, attn_weights = model(x_tensor)

    # Average attention weights across heads and queries -> importance per time step
    attn = attn_weights[0].cpu().numpy()  # (T', T')
    importance = attn.mean(axis=0)  # average over query positions
    importance = importance / importance.max()  # normalize to [0, 1]

    pred_class = cfg.CLASS_NAMES[logits[0].argmax().item()]
    pred_lead = lead_time[0].item()

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={"height_ratios": [3, 1]})

    # Time axis
    if timestamps is not None:
        # Downsample to match attention resolution
        t_ds = np.linspace(0, len(timestamps) - 1, len(importance), dtype=int)
        t_axis = timestamps[t_ds]
    else:
        t_axis = np.arange(len(importance))

    # Plot 1: Input signal with attention overlay
    ax = axes[0]
    input_signal = X_sample[:, 0]  # first feature channel (usually flux)
    t_signal = np.arange(len(input_signal))

    ax.plot(t_signal, input_signal, color="#1a1a5e", linewidth=0.8, alpha=0.8)
    ax.fill_between(t_signal, 0, input_signal, alpha=0.1, color="#1a1a5e")

    # Overlay attention as colored background
    attn_upsampled = np.interp(t_signal, np.linspace(0, len(t_signal), len(importance)),
                                importance)
    for i in range(len(t_signal) - 1):
        ax.axvspan(t_signal[i], t_signal[i+1], alpha=attn_upsampled[i] * 0.5,
                   color="#e65100", linewidth=0)

    ax.set_ylabel("Flux (counts/s)", fontsize=12)
    ax.set_title(f"Prediction: {pred_class}-class flare | Lead time: {pred_lead:.0f} min",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2)

    # Plot 2: Attention importance bar
    ax2 = axes[1]
    ax2.bar(range(len(importance)), importance, color="#e65100", alpha=0.8, width=1.0)
    ax2.set_xlabel("Time step (downsampled)", fontsize=12)
    ax2.set_ylabel("Attention", fontsize=12)
    ax2.set_title("Model Attention - Where is the model looking?", fontsize=12)
    ax2.grid(True, alpha=0.2)

    fig.suptitle("Explainable AI - Attention Heatmap",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_lead_time_distribution(lead_times: np.ndarray,
                                 save_path: Optional[str] = None) -> plt.Figure:
    """Plot distribution of lead times for correct predictions."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(lead_times, bins=30, color="#e65100", alpha=0.8, edgecolor="white")
    ax.axvline(np.mean(lead_times), color="#1a1a5e", linestyle="--", linewidth=2,
               label=f"Mean: {np.mean(lead_times):.1f} min")
    ax.axvline(15, color="#4caf50", linestyle=":", linewidth=2, label="15 min target")
    ax.axvline(30, color="#f44336", linestyle=":", linewidth=2, label="30 min target")

    ax.set_xlabel("Lead Time (minutes)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Flare Prediction Lead Time Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ═══════════════════════════════════════════════════════════════
#  Full Evaluation Pipeline
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def full_evaluation(model: FlareForecaster, X: np.ndarray, y: np.ndarray,
                     lead_times: Optional[np.ndarray] = None,
                     save_dir: Optional[str] = None) -> Dict:
    """
    Run complete evaluation with all metrics and plots.
    """
    device = next(model.parameters()).device
    model.eval()
    save_dir = save_dir or str(cfg.PLOTS_DIR)
    os.makedirs(save_dir, exist_ok=True)

    # Get predictions
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    batch_size = 64
    all_logits = []
    all_lead_preds = []

    for i in range(0, len(X_tensor), batch_size):
        batch = X_tensor[i:i+batch_size]
        logits, lt, _ = model(batch)
        all_logits.append(logits.cpu())
        all_lead_preds.append(lt.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_lead_preds = torch.cat(all_lead_preds, dim=0)

    y_prob = torch.softmax(all_logits, dim=1).numpy()
    y_pred = all_logits.argmax(dim=1).numpy()
    lt_pred = all_lead_preds.squeeze().numpy()

    # Compute metrics
    metrics = compute_all_metrics(y, y_pred, y_prob, lead_times, lt_pred)
    print_metrics(metrics)

    # Generate plots
    print("\nGenerating evaluation plots...")
    plot_confusion_matrix(y, y_pred,
                          save_path=os.path.join(save_dir, "confusion_matrix.png"))
    plot_roc_curves(y, y_prob,
                     save_path=os.path.join(save_dir, "roc_curves.png"))

    if lead_times is not None:
        correct_flare = (y == y_pred) & (y > 0)
        if correct_flare.sum() > 0:
            plot_lead_time_distribution(
                lead_times[correct_flare],
                save_path=os.path.join(save_dir, "lead_time_dist.png")
            )

    # Attention heatmap for a sample
    if len(X) > 0:
        # Find a sample with a flare prediction
        flare_samples = np.where(y > 0)[0]
        if len(flare_samples) > 0:
            sample_idx = flare_samples[0]
            plot_attention_heatmap(
                model, X[sample_idx],
                save_path=os.path.join(save_dir, "attention_heatmap.png")
            )

    print(f"\nPlots saved to {save_dir}/")
    return metrics
