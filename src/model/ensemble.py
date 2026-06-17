"""
Solar Flare Early Warning System - Ensemble Forecasting
Combines multiple models with temperature scaling and confidence thresholds
for NASA-grade prediction reliability.

Components:
  1. TemperatureScaling - Calibrates probabilities to be meaningful
  2. ConfidenceThresholds - Only alerts when confidence is high
  3. EnsembleForecaster - Averages 5 models for robust predictions
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict
from scipy.optimize import minimize_scalar

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config as cfg
from src.model.architecture import FlareForecaster, FlareForecasterLoss
from src.model.augmentation import augment_dataset


class TemperatureScaling:
    """
    Post-hoc temperature scaling for probability calibration.
    
    After training, neural network probabilities are often overconfident.
    Temperature scaling divides logits by a learned temperature T > 1
    before softmax, producing better-calibrated probabilities.
    
    A well-calibrated model means: when it says "70% chance of M-class",
    M-class flares actually occur 70% of the time.
    """
    
    def __init__(self):
        self.temperature = 1.0
    
    def fit(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """
        Find optimal temperature that minimizes negative log-likelihood.
        
        Args:
            logits: (N, n_classes) raw model outputs
            labels: (N,) true class indices
        
        Returns:
            Optimal temperature value
        """
        def nll_loss(T):
            scaled = logits / T
            # Numerically stable softmax
            max_vals = np.max(scaled, axis=1, keepdims=True)
            exp_scaled = np.exp(scaled - max_vals)
            probs = exp_scaled / exp_scaled.sum(axis=1, keepdims=True)
            # NLL
            correct_probs = probs[np.arange(len(labels)), labels]
            correct_probs = np.clip(correct_probs, 1e-10, 1.0)
            return -np.mean(np.log(correct_probs))
        
        result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method='bounded')
        self.temperature = result.x
        print(f"  Temperature scaling: T={self.temperature:.3f} "
              f"(NLL: {result.fun:.4f})")
        return self.temperature
    
    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling and return calibrated probabilities."""
        scaled = logits / self.temperature
        max_vals = np.max(scaled, axis=1, keepdims=True)
        exp_scaled = np.exp(scaled - max_vals)
        return exp_scaled / exp_scaled.sum(axis=1, keepdims=True)


class ConfidenceThresholds:
    """
    Confidence-based prediction with tiered alert levels.
    
    Instead of always predicting the argmax class, only predict
    a flare if the model is sufficiently confident. This dramatically
    reduces false positives.
    
    Alert levels:
      GREEN  - No significant flare expected (confidence < threshold)
      YELLOW - Possible flare (0.4 < confidence < 0.7)
      RED    - High-confidence flare warning (confidence > 0.7)
    """
    
    def __init__(self, flare_threshold: float = 0.5,
                 yellow_threshold: float = 0.4,
                 red_threshold: float = 0.7):
        self.flare_threshold = flare_threshold
        self.yellow_threshold = yellow_threshold
        self.red_threshold = red_threshold
    
    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Predict class with confidence thresholding.
        Only predicts a flare class if max non-None probability > threshold.
        """
        N = len(probabilities)
        predictions = np.zeros(N, dtype=np.int64)
        
        for i in range(N):
            probs = probabilities[i]
            # Get max flare probability (classes 1-4, excluding None=0)
            flare_probs = probs[1:]
            max_flare_prob = flare_probs.max()
            
            if max_flare_prob >= self.flare_threshold:
                predictions[i] = 1 + flare_probs.argmax()  # +1 because class 0 is None
            else:
                predictions[i] = 0  # No flare
        
        return predictions
    
    def predict_with_confidence(self, probabilities: np.ndarray) -> List[Dict]:
        """
        Returns predictions with confidence and alert level.
        """
        results = []
        for i in range(len(probabilities)):
            probs = probabilities[i]
            flare_probs = probs[1:]
            max_flare_prob = float(flare_probs.max())
            max_flare_class = int(1 + flare_probs.argmax())
            
            if max_flare_prob >= self.red_threshold:
                alert = "RED"
                pred_class = max_flare_class
            elif max_flare_prob >= self.yellow_threshold:
                alert = "YELLOW"
                pred_class = max_flare_class
            else:
                alert = "GREEN"
                pred_class = 0
            
            results.append({
                "predicted_class": pred_class,
                "class_name": cfg.CLASS_NAMES[pred_class],
                "confidence": max_flare_prob,
                "alert_level": alert,
                "probabilities": probs.tolist(),
            })
        
        return results


class EnsembleForecaster:
    """
    Ensemble of N FlareForecaster models for robust predictions.
    
    Each model is trained with a different random seed, producing
    diverse predictions. Averaging across models:
    - Reduces variance (more stable predictions)
    - Improves accuracy by 3-10%
    - Provides uncertainty estimates (disagreement = uncertainty)
    
    Combined with temperature scaling and confidence thresholds,
    this produces NASA-grade prediction reliability.
    """
    
    def __init__(self, n_models: int = 5, n_features: int = 9):
        self.n_models = n_models
        self.n_features = n_features
        self.models: List[FlareForecaster] = []
        self.calibrator = TemperatureScaling()
        self.thresholds = ConfidenceThresholds()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_calibrated = False
    
    def train_ensemble(self, X: np.ndarray, y: np.ndarray,
                       lead_times: np.ndarray,
                       epochs: int = 30,
                       augment: bool = True,
                       aug_multiplier: int = 5) -> Dict:
        """
        Train N models with different seeds and data splits.
        
        Uses k-fold-like approach: each model sees slightly different
        data through augmentation with different seeds.
        """
        print(f"\n{'='*70}")
        print(f"  ENSEMBLE TRAINING: {self.n_models} models")
        print(f"{'='*70}")
        
        self.models = []
        all_val_metrics = []
        
        for model_idx in range(self.n_models):
            print(f"\n--- Model {model_idx+1}/{self.n_models} (seed={model_idx*42}) ---")
            
            # Set seed for reproducibility
            seed = model_idx * 42
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Augment with different seed per model
            if augment and len(X) < 2000:
                X_aug, y_aug, lt_aug = augment_dataset(
                    X, y, lead_times,
                    multiplier=aug_multiplier,
                    seed=seed
                )
                print(f"  Augmented: {len(X)} -> {len(X_aug)} samples")
            else:
                X_aug, y_aug, lt_aug = X, y, lead_times
            
            # Shuffle with this seed
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(X_aug))
            X_aug = X_aug[perm]
            y_aug = y_aug[perm]
            lt_aug = lt_aug[perm]
            
            # Train/val split
            n_val = max(int(len(X_aug) * 0.15), 10)
            n_train = len(X_aug) - n_val
            
            X_train = torch.tensor(X_aug[:n_train], dtype=torch.float32)
            y_train = torch.tensor(y_aug[:n_train], dtype=torch.long)
            lt_train = torch.tensor(lt_aug[:n_train], dtype=torch.float32)
            X_val = torch.tensor(X_aug[n_train:], dtype=torch.float32)
            y_val = torch.tensor(y_aug[n_train:], dtype=torch.long)
            lt_val = torch.tensor(lt_aug[n_train:], dtype=torch.float32)
            
            # Create model
            model = FlareForecaster(n_input_channels=self.n_features).to(self.device)
            criterion = FlareForecasterLoss().to(self.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=1e-5
            )
            
            batch_size = 32
            best_val_loss = float("inf")
            patience_counter = 0
            best_state = None
            
            for epoch in range(1, epochs + 1):
                # Train
                model.train()
                total_loss = 0
                correct = 0
                total = 0
                
                indices = torch.randperm(n_train)
                for i in range(0, n_train, batch_size):
                    idx = indices[i:i+batch_size]
                    xb = X_train[idx].to(self.device)
                    yb = y_train[idx].to(self.device)
                    lb = lt_train[idx].to(self.device)
                    
                    optimizer.zero_grad()
                    logits, lt_pred, _ = model(xb)
                    losses = criterion(logits, lt_pred, yb, lb)
                    losses["total"].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    total_loss += losses["total"].item() * len(idx)
                    correct += (logits.argmax(1) == yb).sum().item()
                    total += len(idx)
                
                scheduler.step()
                
                # Validate
                model.eval()
                with torch.no_grad():
                    val_logits, val_lt, _ = model(X_val.to(self.device))
                    val_losses = criterion(val_logits, val_lt,
                                          y_val.to(self.device),
                                          lt_val.to(self.device))
                    val_loss = val_losses["total"].item()
                    val_acc = (val_logits.argmax(1) == y_val.to(self.device)).float().mean().item()
                
                train_loss = total_loss / total
                train_acc = correct / total
                
                if epoch % 5 == 0 or epoch == 1:
                    print(f"  Epoch {epoch:3d}: train_loss={train_loss:.3f} "
                          f"val_loss={val_loss:.3f} val_acc={val_acc:.3f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                
                if patience_counter >= 10:
                    print(f"  Early stopping at epoch {epoch}")
                    break
            
            # Load best model
            if best_state:
                model.load_state_dict(best_state)
            model.eval()
            self.models.append(model)
            
            all_val_metrics.append({
                "best_val_loss": best_val_loss,
                "final_val_acc": val_acc,
            })
        
        # Calibrate on the original (non-augmented) data
        print(f"\n--- Calibrating ensemble ---")
        self._calibrate(X, y)
        
        return {"per_model": all_val_metrics}
    
    def _calibrate(self, X: np.ndarray, y: np.ndarray):
        """Calibrate temperature scaling on validation data."""
        logits = self._get_ensemble_logits(X)
        self.calibrator.fit(logits, y)
        self.is_calibrated = True
    
    def _get_ensemble_logits(self, X: np.ndarray) -> np.ndarray:
        """Get averaged logits from all models."""
        all_logits = []
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                # Process in batches to avoid OOM
                batch_logits = []
                for i in range(0, len(X), 64):
                    batch = X_tensor[i:i+64]
                    logits, _, _ = model(batch)
                    batch_logits.append(logits.cpu().numpy())
                all_logits.append(np.concatenate(batch_logits, axis=0))
        
        return np.mean(all_logits, axis=0)
    
    def _get_ensemble_lead_times(self, X: np.ndarray) -> np.ndarray:
        """Get averaged lead time predictions from all models."""
        all_lts = []
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                batch_lts = []
                for i in range(0, len(X), 64):
                    batch = X_tensor[i:i+64]
                    _, lt, _ = model(batch)
                    batch_lts.append(lt.cpu().numpy())
                all_lts.append(np.concatenate(batch_lts, axis=0))
        
        result = np.mean(all_lts, axis=0).squeeze()
        # Ensure 1-d even for single sample
        return np.atleast_1d(result)
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get calibrated predictions with confidence thresholds.
        
        Returns:
            predictions: (N,) class indices
            probabilities: (N, n_classes) calibrated probabilities
        """
        logits = self._get_ensemble_logits(X)
        
        if self.is_calibrated:
            probs = self.calibrator.calibrate(logits)
        else:
            # Standard softmax
            exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = exp_l / exp_l.sum(axis=1, keepdims=True)
        
        predictions = self.thresholds.predict(probs)
        return predictions, probs
    
    def predict_detailed(self, X: np.ndarray) -> List[Dict]:
        """Get detailed predictions with alert levels."""
        logits = self._get_ensemble_logits(X)
        lead_times = self._get_ensemble_lead_times(X)
        
        if self.is_calibrated:
            probs = self.calibrator.calibrate(logits)
        else:
            exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = exp_l / exp_l.sum(axis=1, keepdims=True)
        
        detailed = self.thresholds.predict_with_confidence(probs)
        for i, d in enumerate(detailed):
            d["lead_time_min"] = float(lead_times[i]) if i < len(lead_times) else 0.0
        
        return detailed
    
    def get_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction uncertainty (disagreement between models).
        High uncertainty = models disagree = less reliable prediction.
        """
        all_preds = []
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                batch_preds = []
                for i in range(0, len(X), 64):
                    batch = X_tensor[i:i+64]
                    logits, _, _ = model(batch)
                    probs = F.softmax(logits, dim=1)
                    batch_preds.append(probs.cpu().numpy())
                all_preds.append(np.concatenate(batch_preds, axis=0))
        
        # Uncertainty = std of predictions across models
        preds_stack = np.stack(all_preds, axis=0)  # (n_models, N, n_classes)
        return preds_stack.std(axis=0).mean(axis=1)  # (N,)
    
    def save(self, save_dir: str):
        """Save all ensemble models and calibration parameters."""
        os.makedirs(save_dir, exist_ok=True)
        for i, model in enumerate(self.models):
            path = os.path.join(save_dir, f"ensemble_model_{i}.pt")
            torch.save(model.state_dict(), path)
        
        # Save calibration
        cal_path = os.path.join(save_dir, "calibration.npz")
        np.savez(cal_path,
                 temperature=self.calibrator.temperature,
                 flare_threshold=self.thresholds.flare_threshold)
        print(f"  Saved {len(self.models)} ensemble models to {save_dir}")
    
    def load(self, save_dir: str):
        """Load ensemble models and calibration."""
        self.models = []
        for i in range(self.n_models):
            path = os.path.join(save_dir, f"ensemble_model_{i}.pt")
            if os.path.exists(path):
                model = FlareForecaster(n_input_channels=self.n_features).to(self.device)
                model.load_state_dict(
                    torch.load(path, map_location=self.device, weights_only=True)
                )
                model.eval()
                self.models.append(model)
        
        cal_path = os.path.join(save_dir, "calibration.npz")
        if os.path.exists(cal_path):
            cal = np.load(cal_path)
            self.calibrator.temperature = float(cal["temperature"])
            self.thresholds.flare_threshold = float(cal["flare_threshold"])
            self.is_calibrated = True
        
        print(f"  Loaded {len(self.models)} ensemble models from {save_dir}")
