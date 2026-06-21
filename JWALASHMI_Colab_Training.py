"""
JWALASHMI v5.3 - Google Colab Training Script
Run this in Google Colab with free GPU (T4)

Instructions:
1. Go to https://colab.research.google.com/
2. File > Upload Notebook > Upload this as .py or paste in cells
3. Runtime > Change runtime type > GPU (T4)
4. Run all cells

This script:
- Clones your repo from GitHub
- Trains 10-model ensemble on Colab GPU
- Saves results back to the repo
"""

# ============================================================
# CELL 1: Setup
# ============================================================
import subprocess
import os

# Clone repo
if not os.path.exists('/content/ISRO'):
    subprocess.run(['git', 'clone', 'https://github.com/FrozenLionMax/Jwalashmi.git', '/content/ISRO'], check=True)

os.chdir('/content/ISRO')

# Install dependencies
subprocess.run(['pip', 'install', '-q', 'astropy', 'scikit-learn', 'tqdm'], check=True)

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# CELL 2: Run Pipeline
# ============================================================
import sys
sys.path.insert(0, '/content/ISRO')

# Set unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

# Run the v5 pipeline
exec(open('run_v5_max_accuracy.py').read())
