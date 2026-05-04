import torch
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.dataset.pandas import PandasDataset
import pandas as pd
import numpy as np

# Test basic imports
print("Imports successful!")
print(f"PyTorch version: {torch.__version__}")

# Create sample time series data
dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
values = np.random.randn(100).cumsum() + 100

df = pd.DataFrame({'target': values}, index=dates)
dataset = PandasDataset(df)

print(f"Dataset created with {len(df)} observations")
print("Test passed!")
