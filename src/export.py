import os
import shutil

# Paths
ARTIFACTS_DIR = "artifacts"
MODEL_FILENAME = "rs.joblib"
SOURCE_PATH = os.path.join(ARTIFACTS_DIR, MODEL_FILENAME)
DEST_DIR = "models"
DEST_PATH = os.path.join(DEST_DIR, MODEL_FILENAME)

# Ensure model exists
if not os.path.exists(SOURCE_PATH):
    raise FileNotFoundError(f"Trained model not found at: {SOURCE_PATH}")

# Create destination folder if not exists
os.makedirs(DEST_DIR, exist_ok=True)

# Copy model to destination
shutil.copy(SOURCE_PATH, DEST_PATH)

print(f" Model exported to: {DEST_PATH}")