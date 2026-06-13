import time
from io import BytesIO

import streamlit as st
from PIL import Image

from config import APP_SUBTITLE, APP_TITLE, SUPPORTED_IMAGE_TYPES
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
    .block-container {
        padding-top: 1.0rem;
        padding-bottom: 1.0rem;
        max-width: 1500px;
    }

    .title-wrap {
        padding-bottom: 0.35rem;
    }

    .small-note {
        font-size: 0.92rem;
        color: #475569;
        line-height: 1.45;
    }

    .field-label {
        font-size: 0.92rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 0.35rem;
    }

    .section-title {
        margin-top: 0.15rem;
        margin-bottom: 0.75rem;
        font-size: 1.35rem;
        font-weight: 700;
        color: #0f172a;
    }

    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 16px 14px;
        min-height: 116px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }

    .metric-label {
        font-size: 0.88rem;
        color: #334155;
        margin-bottom: 8px;
        line-height: 1.2;
    }

    .metric-value {
        font-size: 1.75rem;
        font-weight: 500;
        color: #0f172a;
        line-height: 1.15;
        word-break: break-word;
    }

    .metric-fps {
        display: inline-block;
        margin-top: 8px;
        padding: 3px 8px;
        border-radius: 999px;
        background: #dcfce7;
        color: #166534;
        font-size: 0.82rem;
        font-weight: 500;
    }

    .image-spacer {
        height: 0.4rem;
    }

    .probability-section {
        margin-top: 1.1rem;
    }

    .right-lower-spacer {
        height: 0.15rem;
    }

    div[data-testid="stProgress"] > div > div > div > div {
        background-color: #2563eb;
    }
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


def show_prediction_cards(result, inference_ms, fps):
    st.markdown(
        "<div class='section-title'>Prediction and attention visualization</div>",
        unsafe_allow_html=True,
    )

    metric_1, metric_2, metric_3 = st.columns(3, gap="medium")

    with metric_1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Predicted class</div>
                <div class="metric-value">{result["pred_label"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with metric_2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Confidence</div>
                <div class="metric-value">{result["confidence"] * 100:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with metric_3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Inference speed</div>
                <div class="metric-value">{inference_ms:.1f} ms</div>
                <div class="metric-fps">↑ {fps:.2f} FPS</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


st.markdown(
    f"""
    <div class='title-wrap'>
      <h1 style='margin-bottom:0'>🌹 {APP_TITLE}</h1>
      <p style='margin-top:0.25rem; color:#475569'>
        {APP_SUBTITLE} with attention heatmap
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


left_col, right_col = st.columns(
    [0.40, 0.60],
    gap="large",
    vertical_alignment="top",
)

image = None


with left_col:
    input_source_col, upload_col = st.columns(
        [0.38, 0.62],
        gap="medium",
        vertical_alignment="top",
    )

    with input_source_col:
        st.markdown(
            "<div class='field-label'>Image source</div>",
            unsafe_allow_html=True,
        )

        input_mode = st.radio(
            "Choose image source",
            ["Upload image", "Camera snapshot"],
            label_visibility="collapsed",
        )

    with upload_col:
        if input_mode == "Upload image":
            st.markdown(
                "<div class='field-label'>Upload image</div>",
                unsafe_allow_html=True,
            )

            image_source = st.file_uploader(
                "Upload a rose leaf image",
                type=SUPPORTED_IMAGE_TYPES,
                accept_multiple_files=False,
                label_visibility="collapsed",
            )

        else:
            st.markdown(
                "<div class='field-label'>Camera snapshot</div>",
                unsafe_allow_html=True,
            )

            image_source = st.camera_input(
                "Capture a rose leaf image",
                label_visibility="collapsed",
            )

    st.markdown(
        "<div class='small-note'>The prediction starts automatically after an image is provided.</div>",
        unsafe_allow_html=True,
    )

    if image_source is not None:
        try:
            image = open_uploaded_image(image_source)

            st.markdown(
                "<div class='image-spacer'></div>",
                unsafe_allow_html=True,
            )

            st.image(
                image,
                caption="Selected input image",
                use_container_width=True,
            )

        except Exception as exc:
            st.error(f"Could not read the image: {exc}")
            image = None


if image is None:
    with right_col:
        st.markdown(
            "<div class='section-title'>Prediction and attention visualization</div>",
            unsafe_allow_html=True,
        )

        st.info(
            "Upload or capture a leaf image to view the predicted disease, "
            "confidence, class probabilities, and attention overlay."
        )

    st.stop()


try:
    with right_col:
        with st.spinner("Loading model and running inference..."):
            model, processor, device, model_path = cached_load_model_bundle()

            start = time.perf_counter()
            result = predict_image(image, model, processor, device)
            inference_ms = (time.perf_counter() - start) * 1000.0
            fps = 1000.0 / inference_ms if inference_ms > 0 else 0.0


except FileNotFoundError as exc:
    with right_col:
        st.error(str(exc))
        st.markdown(
            "Place your trained notebook weight file, for example "
            "`best_dspa_clip_model.pth`, inside the `models/` folder and rerun the app."
        )

    st.stop()


except Exception as exc:
    with right_col:
        st.exception(exc)

    st.stop()


with right_col:
    show_prediction_cards(
        result=result,
        inference_ms=inference_ms,
        fps=fps,
    )

    st.caption(f"Model weights: `{model_path.name}` | Device: `{device}`")

    st.markdown(
        "<div class='right-lower-spacer'></div>",
        unsafe_allow_html=True,
    )

    viz_col_1, viz_col_2 = st.columns(
        2,
        gap="medium",
        vertical_alignment="top",
    )

    with viz_col_1:
        st.image(
            result["visuals"]["heatmap"],
            caption="Attention heatmap",
            use_container_width=True,
        )

    with viz_col_2:
        st.image(
            result["visuals"]["overlay"],
            caption="Attention overlay on input",
            use_container_width=True,
        )

    st.markdown("**Interpretation note**")

    st.markdown(
        f"""
        <div class='small-note'>
            {DISEASE_NOTES.get(
                result['pred_label'],
                'Review the attention overlay before making a final decision.'
            )}
        </div>
        """,
        unsafe_allow_html=True,
    )


with left_col:
    st.markdown(
        "<div class='probability-section'></div>",
        unsafe_allow_html=True,
    )

    st.markdown("**Class probabilities**")

    for row in result["probabilities"]:
        st.progress(
            min(max(row["Probability"], 0.0), 1.0),
            text=f"{row['Class']}: {row['Probability'] * 100:.2f}%",
        )
