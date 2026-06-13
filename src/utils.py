from typing import Dict

DISEASE_NOTES: Dict[str, str] = {
    "Black Spot": (
        "The model found visual evidence consistent with dark circular or irregular spot symptoms. "
        "Use the attention overlay to verify whether the highlighted regions correspond to visible lesions."
    ),
    "Dry Leaf": (
        "The model found visual evidence consistent with drying, browning, curling, or brittle leaf tissue. "
        "Check whether the highlighted area is truly a stressed/dry region rather than background."
    ),
    "Healthy Leaf": (
        "The model found the strongest evidence for normal green leaf appearance without obvious disease or damage. "
        "Still inspect the attention map, because early symptoms can be visually subtle."
    ),
    "Leaf Holes": (
        "The model found visual evidence consistent with holes, missing tissue, or pest-like damage. "
        "Confirm whether the highlighted patches correspond to actual perforations or torn areas."
    ),
}


def compact_label(label: str) -> str:
    return label.replace("_", " ").strip()
