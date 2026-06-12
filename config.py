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

# Place your trained PyTorch weights in /models. The app checks these paths in order.
MODEL_CANDIDATES = [
    BASE_DIR / "models" / "best_dspa_clip_model.pth"
]

SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png", "webp"]
