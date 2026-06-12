# Rose Leaf Disease Detection Streamlit App

This Streamlit app deploys the trained **Disease-Semantic Patch Attention CLIP** model from the supplied notebook. It provides disease prediction, confidence scores, class probabilities, and the model's own class-conditioned patch attention heatmap.

## Project structure

```text
dspa_roseleaf_streamlit/
├── app.py
├── config.py
├── prompts.py
├── requirements.txt
├── README.md
├── models/
│   └── best_dspa_clip_model.pth      # add your trained weights here
├── sample_images/
└── src/
    ├── __init__.py
    ├── inference.py
    ├── model.py
    └── utils.py
```

## Setup

1. Copy the trained model weight file from your notebook output into the `models/` folder.

   Recommended file name:

   ```text
   models/best_dspa_clip_model.pth
   ```

   The app also checks for:

   ```text
   models/best_text_guided_clip_model.pth
   best_dspa_clip_model.pth
   best_text_guided_clip_model.pth
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   streamlit run app.py
   ```

## Important notes

- Keep the class order in `config.py` exactly the same as the training notebook:

  ```python
  ["Black Spot", "Dry Leaf", "Healthy Leaf", "Leaf Holes"]
  ```

- This app visualizes `attention_weights` returned by `DiseaseSemanticPatchAttentionCLIP(..., return_attention=True)`. Therefore, the heatmap is not a post-hoc Grad-CAM/LIME visualization. It is the proposed disease-semantic patch attention used in the model's forward pass.

- The first run may take time because Hugging Face downloads `openai/clip-vit-base-patch32`. After that, Streamlit caches the loaded model.

## Why this interface is useful for a journal paper

The interface demonstrates the full research pipeline in an applied setting:

1. A leaf image is provided through upload or camera snapshot.
2. The trained CLIP-based disease-semantic model predicts the disease class.
3. The model returns class-conditioned patch attention.
4. The app overlays the attention on the input image, showing which regions supported the disease prediction.
5. Confidence and probability scores are shown beside the visualization, reducing the need for scrolling during presentation.

