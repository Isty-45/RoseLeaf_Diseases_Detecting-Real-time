import time
from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image

from config import APP_SUBTITLE, APP_TITLE, CLASS_NAMES, SUPPORTED_IMAGE_TYPES
from src.inference import load_model_bundle, predict_image
from src.utils import DISEASE_NOTES


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🌹",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.0rem; padding-bottom: 1.0rem; max-width: 1500px;}
    div[data-testid="stMetric"] {background: #f8fafc; border: 1px solid #e5e7eb; padding: 12px; border-radius: 14px;}
    .small-note {font-size: 0.92rem; color: #475569; line-height: 1.45;}
    .title-wrap {padding-bottom: 0.25rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def cached_load_model_bundle():
    return load_model_bundle()


def open_uploaded_image(uploaded_file) -> Image.Image:
    image_bytes = uploaded_file.getvalue()
    return Image.open(BytesIO(image_bytes)).convert("RGB")


st.markdown(
    f"""
    <div class='title-wrap'>
      <h1 style='margin-bottom:0'>🌹 {APP_TITLE}</h1>
      <p style='margin-top:0.25rem; color:#475569'>{APP_SUBTITLE} with class-conditioned attention heatmap</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([0.42, 0.58], gap="large", vertical_alignment="top")

with left_col:
    st.subheader("Input")
    input_mode = st.radio(
        "Choose image source",
        ["Upload image", "Camera snapshot"],
        horizontal=True,
        label_visibility="collapsed",
    )

    image_source = None
    if input_mode == "Upload image":
        image_source = st.file_uploader(
            "Upload a rose leaf image",
            type=SUPPORTED_IMAGE_TYPES,
            accept_multiple_files=False,
        )
    else:
        image_source = st.camera_input("Capture a rose leaf image")

    st.markdown("<div class='small-note'>The prediction starts automatically after an image is provided.</div>", unsafe_allow_html=True)

    image = None
    if image_source is not None:
        try:
            image = open_uploaded_image(image_source)
            st.image(image, caption="Selected input image", use_container_width=True)
        except Exception as exc:
            st.error(f"Could not read the image: {exc}")

    with st.expander("Model details", expanded=False):
        st.write("**Classes:** " + ", ".join(CLASS_NAMES))
        st.write("**Visualization:** predicted-class disease-semantic patch attention, not Grad-CAM/LIME.")

with right_col:
    st.subheader("Prediction and attention visualization")

    if image is None:
        st.info("Upload or capture a leaf image to view the predicted disease, confidence, probabilities, and attention overlay.")
        st.stop()

    try:
        with st.spinner("Loading model and running inference..."):
            model, processor, device, model_path = cached_load_model_bundle()
            start = time.perf_counter()
            result = predict_image(image, model, processor, device)
            inference_ms = (time.perf_counter() - start) * 1000.0
            fps = 1000.0 / inference_ms if inference_ms > 0 else 0.0
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.markdown(
            "Place your trained notebook weight file, for example `best_dspa_clip_model.pth`, inside the `models/` folder and rerun the app."
        )
        st.stop()
    except Exception as exc:
        st.exception(exc)
        st.stop()

    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Predicted class", result["pred_label"])
    metric_2.metric("Confidence", f"{result['confidence'] * 100:.2f}%")
    metric_3.metric("Inference speed", f"{inference_ms:.1f} ms", f"{fps:.2f} FPS")

    st.caption(f"Model weights: `{model_path.name}` | Device: `{device}`")

    viz_col_1, viz_col_2 = st.columns(2, gap="medium", vertical_alignment="top")
    with viz_col_1:
        st.image(result["visuals"]["heatmap"], caption="Attention heatmap", use_container_width=True)
    with viz_col_2:
        st.image(result["visuals"]["overlay"], caption="Attention overlay on input", use_container_width=True)

    prob_col, note_col = st.columns([0.52, 0.48], gap="medium", vertical_alignment="top")
    with prob_col:
        st.markdown("**Class probabilities**")
        for row in result["probabilities"]:
            st.progress(
                min(max(row["Probability"], 0.0), 1.0),
                text=f"{row['Class']}: {row['Probability'] * 100:.2f}%",
            )

    with note_col:
        st.markdown("**Interpretation note**")
        st.markdown(
            f"<div class='small-note'>{DISEASE_NOTES.get(result['pred_label'], 'Review the attention overlay before making a final decision.')}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='small-note'><br>This interface visualizes the same class-conditioned patch attention used by the model, so it is more aligned with the proposed method than a generic post-hoc Grad-CAM section.</div>",
            unsafe_allow_html=True,
        )

    prob_df = pd.DataFrame(result["probabilities"])
    prob_df["Probability (%)"] = (prob_df["Probability"] * 100).round(2)
    prob_df = prob_df.drop(columns=["Probability"])

    with st.expander("Probability table", expanded=False):
        st.dataframe(prob_df, use_container_width=True, hide_index=True)
