from pathlib import Path
import os, torch

SEED = 30
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"

for directory in [DATA_DIR, OUTPUTS_DIR, FIGURES_DIR, MODELS_DIR]:
    if not directory.exists(): directory.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")