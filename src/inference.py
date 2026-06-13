from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from transformers import CLIPModel, CLIPProcessor

from config import (
    ALPHA_GLOBAL,
    ATTN_DIM,
    CLASS_NAMES,
    CLIP_MODEL_NAME,
    DROPOUT,
    IMAGE_SIZE,
    MODEL_CANDIDATES,
    MODEL_DRIVE_FILE_ID,
    MODEL_DRIVE_URL,
    MODEL_PATH,
    AUTO_DOWNLOAD_MODEL,
    UNFREEZE_LAST_N_VISION_BLOCKS,
)
from prompts import TEXT_PROMPTS
from src.model import DiseaseSemanticPatchAttentionCLIP, build_prompt_ensemble_prototypes


MIN_EXPECTED_MODEL_BYTES = 1_000_000


def _is_valid_model_file(path: Path) -> bool:
    """
    Check whether a candidate file is likely to be a real PyTorch checkpoint.
    This prevents the app from trying to load an incomplete Google Drive download
    or an HTML error page as a .pth file.
    """
    if not path.exists() or not path.is_file():
        return False

    if path.stat().st_size < MIN_EXPECTED_MODEL_BYTES:
        return False

    try:
        with path.open("rb") as file:
            header = file.read(512).lower()
        if b"<html" in header or b"<!doctype html" in header:
            return False
    except Exception:
        return False

    return True


def _download_model_from_google_drive() -> Path:
    """
    Download the trained model from Google Drive when it is not present locally.

    Important:
    The Google Drive file must be shared as:
    Anyone with the link -> Viewer
    """
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists() and not _is_valid_model_file(MODEL_PATH):
        try:
            MODEL_PATH.unlink()
        except Exception:
            pass

    try:
        import gdown
    except ImportError as exc:
        raise ImportError(
            "The model file is missing and gdown is not installed. "
            "Install dependencies with `pip install -r requirements.txt`, "
            f"or place the model manually at: {MODEL_PATH}"
        ) from exc

    direct_url = f"https://drive.google.com/uc?export=download&id={MODEL_DRIVE_FILE_ID}"

    downloaded_path = None

    try:
        downloaded_path = gdown.download(
            id=MODEL_DRIVE_FILE_ID,
            output=str(MODEL_PATH),
            quiet=False,
        )
    except TypeError:
        downloaded_path = gdown.download(
            url=direct_url,
            output=str(MODEL_PATH),
            quiet=False,
        )
    except Exception:
        downloaded_path = gdown.download(
            url=direct_url,
            output=str(MODEL_PATH),
            quiet=False,
        )

    if downloaded_path is None:
        raise FileNotFoundError(
            "Automatic model download failed. The Google Drive file may not be publicly accessible. "
            "Set the model file permission to 'Anyone with the link can view'. "
            f"Drive URL: {MODEL_DRIVE_URL}"
        )

    if not _is_valid_model_file(MODEL_PATH):
        raise FileNotFoundError(
            "The model file was downloaded, but it does not look like a valid PyTorch checkpoint. "
            "This usually happens when Google Drive returns an access-denied HTML page instead of the real .pth file. "
            "Please make sure the Drive file is shared as 'Anyone with the link can view'. "
            f"Expected path: {MODEL_PATH}"
        )

    return MODEL_PATH


def find_model_path() -> Path:
    for path in MODEL_CANDIDATES:
        if _is_valid_model_file(path):
            return path

    if AUTO_DOWNLOAD_MODEL:
        return _download_model_from_google_drive()

    candidate_list = "\n".join(f"- {p}" for p in MODEL_CANDIDATES)
    raise FileNotFoundError(
        "Trained model weights were not found. Place your .pth file in one of these locations:\n"
        f"{candidate_list}"
    )


def _clean_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    """
    Handle checkpoints saved either as raw state_dict or nested checkpoint dictionaries.
    """
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "The loaded checkpoint is not a valid state_dict or checkpoint dictionary."
        )

    cleaned = {}
    for key, value in checkpoint.items():
        if key.startswith("module."):
            key = key.replace("module.", "", 1)
        cleaned[key] = value

    return cleaned


def _load_checkpoint(model_path: Path, device: torch.device):
    """
    PyTorch versions differ in the supported arguments of torch.load.
    This keeps the app compatible across local and Streamlit environments.
    """
    try:
        return torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(model_path, map_location=device)


def load_model_bundle() -> Tuple[DiseaseSemanticPatchAttentionCLIP, CLIPProcessor, torch.device, Path]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    text_prototypes = build_prompt_ensemble_prototypes(
        clip_model=clip_model,
        processor=processor,
        prompt_dict=TEXT_PROMPTS,
        class_names=CLASS_NAMES,
        device=device,
    )

    model = DiseaseSemanticPatchAttentionCLIP(
        clip_model=clip_model,
        text_prototypes=text_prototypes,
        attn_dim=ATTN_DIM,
        dropout=DROPOUT,
        alpha_global=ALPHA_GLOBAL,
        unfreeze_last_n_vision_blocks=UNFREEZE_LAST_N_VISION_BLOCKS,
    ).to(device)

    model_path = find_model_path()
    checkpoint = _load_checkpoint(model_path, device)
    state_dict = _clean_state_dict(checkpoint)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model, processor, device, model_path


def prepare_image(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    return image


def preprocess_for_clip(image: Image.Image, processor: CLIPProcessor, device: torch.device) -> torch.Tensor:
    processed = processor(images=image, return_tensors="pt")
    return processed["pixel_values"].to(device)


def _normalize_map(attn_map: np.ndarray) -> np.ndarray:
    denom = float(attn_map.max() - attn_map.min())
    if denom < 1e-8:
        return np.zeros_like(attn_map, dtype=np.float32)
    return ((attn_map - attn_map.min()) / (denom + 1e-8)).astype(np.float32)


def build_attention_visuals(image: Image.Image, attention_vector: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Convert 49 patch attention weights into:
    1. resized attention map
    2. colored heatmap
    3. heatmap overlay on the original image
    """
    n_patches = int(attention_vector.shape[0])
    grid_size = int(np.sqrt(n_patches))

    if grid_size * grid_size != n_patches:
        raise ValueError(
            f"Cannot reshape {n_patches} attention weights into a square patch grid."
        )

    rgb = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.float32) / 255.0

    attn_map = attention_vector.reshape(grid_size, grid_size)
    attn_map = _normalize_map(attn_map)

    attn_resized = cv2.resize(
        attn_map,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_CUBIC,
    )

    heatmap = cv2.applyColorMap(
        np.uint8(255 * attn_resized),
        cv2.COLORMAP_JET,
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    overlay = np.clip(0.60 * rgb + 0.40 * heatmap, 0, 1)

    return {
        "original": np.uint8(255 * rgb),
        "attention_map": attn_resized,
        "heatmap": np.uint8(255 * heatmap),
        "overlay": np.uint8(255 * overlay),
    }


@torch.inference_mode()
def predict_image(
    image: Image.Image,
    model: DiseaseSemanticPatchAttentionCLIP,
    processor: CLIPProcessor,
    device: torch.device,
    class_names: List[str] = CLASS_NAMES,
) -> Dict:
    image = prepare_image(image)
    pixel_values = preprocess_for_clip(image, processor, device)

    output = model(pixel_values, return_attention=True)

    logits = output["logits"]
    probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_label = class_names[pred_idx]
    confidence = float(probs[pred_idx])

    attention_vector = output["attention_weights"][0, pred_idx].detach().cpu().numpy()
    visuals = build_attention_visuals(image, attention_vector)

    probability_table = [
        {
            "Class": class_names[i],
            "Probability": float(probs[i]),
        }
        for i in range(len(class_names))
    ]

    probability_table = sorted(
        probability_table,
        key=lambda row: row["Probability"],
        reverse=True,
    )

    return {
        "pred_idx": pred_idx,
        "pred_label": pred_label,
        "confidence": confidence,
        "probabilities": probability_table,
        "visuals": visuals,
    }
