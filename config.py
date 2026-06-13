from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

APP_TITLE = "Rose Leaf Disease Detection"
APP_SUBTITLE = "Disease-Semantic Patch Attention CLIP"

# IMPORTANT: keep this class order exactly the same as the order used during training.
# In the supplied notebook, the sorted class order is:
CLASS_NAMES = ["Black Spot", "Dry Leaf", "Healthy Leaf", "Leaf Holes"]

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
IMAGE_SIZE = 224
ATTN_DIM = 256
DROPOUT = 0.20
ALPHA_GLOBAL = 0.30
UNFREEZE_LAST_N_VISION_BLOCKS = 1

# Preferred local path for the trained PyTorch weights.
# The app first checks the local files below. If none is found, it downloads
# the model from the Google Drive file ID provided here.
MODEL_DIR = BASE_DIR / "models"
MODEL_FILENAME = "best_dspa_clip_model.pth"
MODEL_PATH = MODEL_DIR / MODEL_FILENAME

# Google Drive model source.
# Original link:
# https://drive.google.com/file/d/1Zw_vejDRsMJ_KUrThXl3hqbm10rVxJzX/view?usp=drive_link
MODEL_DRIVE_FILE_ID = "1Zw_vejDRsMJ_KUrThXl3hqbm10rVxJzX"
MODEL_DRIVE_URL = f"https://drive.google.com/uc?id={MODEL_DRIVE_FILE_ID}"
AUTO_DOWNLOAD_MODEL = True

# The app checks these paths in order before downloading from Google Drive.
MODEL_CANDIDATES = [
    MODEL_PATH,
    BASE_DIR / "models" / "best_text_guided_clip_model.pth",
    BASE_DIR / "best_dspa_clip_model.pth",
    BASE_DIR / "best_text_guided_clip_model.pth",
]

SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png", "webp"]
