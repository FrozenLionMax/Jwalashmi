"""
JWALASHMI - ONNX Export & Quantization
========================================
Converts PyTorch FlareForecaster to ONNX + INT8 for <10ms inference.

Usage:
    python analysis/export_onnx.py --model models/v6_1_ensemble/model_0.pt

Requirements:
    pip install onnx onnxruntime
"""

import torch
import torch.nn as nn
import numpy as np
import time
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_dummy_model():
    """Create a FlareForecaster-compatible model for export."""
    class FlareForecasterExport(nn.Module):
        def __init__(self, n_features=9, n_classes=5):
            super().__init__()
            self.conv1 = nn.Sequential(
                nn.Conv1d(n_features, 64, 7, padding=3),
                nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4))
            self.conv2 = nn.Sequential(
                nn.Conv1d(64, 128, 5, padding=2),
                nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3), nn.MaxPool1d(4))
            self.conv3 = nn.Sequential(
                nn.Conv1d(128, 256, 3, padding=1),
                nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                nn.AdaptiveAvgPool1d(32))
            self.attn = nn.MultiheadAttention(256, 4, batch_first=True)
            self.norm = nn.LayerNorm(256)
            self.fc = nn.Sequential(
                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5))
            self.classifier = nn.Linear(128, n_classes)
            self.regressor = nn.Linear(128, 1)

        def forward(self, x):
            # x: (batch, time, features) -> (batch, features, time)
            x = x.transpose(1, 2)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            # Attention: (batch, channels, time) -> (batch, time, channels)
            x = x.transpose(1, 2)
            attn_out, _ = self.attn(x, x, x)
            x = self.norm(x + attn_out)
            x = x.mean(dim=1)  # Global average pool
            x = self.fc(x)
            logits = self.classifier(x)
            lead = self.regressor(x)
            return logits, lead

    return FlareForecasterExport()


def export_to_onnx(model, output_path, n_features=9):
    """Export PyTorch model to ONNX format."""
    model.eval()
    dummy_input = torch.randn(1, 3600, n_features)

    torch.onnx.export(
        model, dummy_input, output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['logits', 'lead_time'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'logits': {0: 'batch_size'},
            'lead_time': {0: 'batch_size'}
        }
    )
    print(f"  [OK] ONNX model saved: {output_path}")
    print(f"  [OK] Size: {os.path.getsize(output_path) / 1024:.1f} KB")


def benchmark_inference(model_path, n_runs=100, n_features=9):
    """Benchmark ONNX Runtime inference speed."""
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        dummy = np.random.randn(1, 3600, n_features).astype(np.float32)

        # Warmup
        for _ in range(10):
            session.run(None, {'input': dummy})

        # Benchmark
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            session.run(None, {'input': dummy})
            times.append((time.perf_counter() - start) * 1000)

        times = np.array(times)
        print(f"\n  Inference Latency ({n_runs} runs):")
        print(f"    Mean:   {times.mean():.2f} ms")
        print(f"    Median: {np.median(times):.2f} ms")
        print(f"    P95:    {np.percentile(times, 95):.2f} ms")
        print(f"    P99:    {np.percentile(times, 99):.2f} ms")
        print(f"    Min:    {times.min():.2f} ms")

        target_met = np.median(times) < 10
        print(f"\n  Target (<10ms): {\'PASSED\' if target_met else \'NEEDS GPU/TensorRT\'}}")

        return times.mean()

    except ImportError:
        print("\n  [SKIP] Install onnxruntime: pip install onnxruntime")
        return None


def quantize_model(input_path, output_path):
    """Apply dynamic INT8 quantization for further speedup."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quantize_dynamic(
            input_path, output_path,
            weight_type=QuantType.QInt8
        )
        orig_size = os.path.getsize(input_path)
        quant_size = os.path.getsize(output_path)
        print(f"\n  [OK] Quantized model saved: {output_path}")
        print(f"  [OK] Size reduction: {orig_size/1024:.1f}KB -> {quant_size/1024:.1f}KB ({quant_size/orig_size:.0%})")

    except ImportError:
        print("\n  [SKIP] Install: pip install onnxruntime")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='JWALASHMI ONNX Export')
    parser.add_argument('--model', type=str, default=None, help='Path to .pt model')
    parser.add_argument('--output', type=str, default='models/jwalashmi.onnx')
    parser.add_argument('--features', type=int, default=9)
    parser.add_argument('--benchmark', action='store_true', default=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    print("\n" + "=" * 60)
    print("  JWALASHMI - ONNX Export & Optimization")
    print("=" * 60)

    # Load or create model
    if args.model and os.path.exists(args.model):
        print(f"\n  Loading model: {args.model}")
        model = torch.load(args.model, map_location='cpu')
    else:
        print(f"\n  Creating reference model (use --model to load trained weights)")
        model = create_dummy_model()

    # Export
    print(f"\n--- ONNX Export ---")
    export_to_onnx(model, args.output, args.features)

    # Quantize
    quant_path = args.output.replace('.onnx', '_int8.onnx')
    print(f"\n--- INT8 Quantization ---")
    quantize_model(args.output, quant_path)

    # Benchmark
    if args.benchmark:
        print(f"\n--- Benchmark (FP32) ---")
        benchmark_inference(args.output, n_features=args.features)

        if os.path.exists(quant_path):
            print(f"\n--- Benchmark (INT8) ---")
            benchmark_inference(quant_path, n_features=args.features)

    print(f"\n  Done! Models saved to: {args.output}")
