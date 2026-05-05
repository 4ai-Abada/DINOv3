import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from huggingface_hub import login

if "HF_TOKEN" in st.secrets:
    login(token=st.secrets["HF_TOKEN"])

# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cleanliness Classifier",
    page_icon="🧼",
    layout="wide",
)

st.title("🧼 Cleanliness Classifier")
st.caption("DINOv3-powered clean/dirty detector")

# ─────────────────────────────────────────────────────────
# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    HEAD_PATH = st.text_input("Model path", "classifier_head.pth")
    st.divider()
    st.caption("💡 Upload one or more images to classify them")

# ─────────────────────────────────────────────────────────
# Model definition (must match training)
class DINOv3Classifier(nn.Module):
    def __init__(self, model_name, hidden_size, dropout=0.3, num_classes=2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, pixel_values):
        with torch.no_grad():
            out = self.backbone(pixel_values=pixel_values)
        return self.head(out.pooler_output)


@st.cache_resource
def load_model(head_path):
    """Load DINOv3 backbone + trained head. Cached so it loads once."""
    ckpt = torch.load(head_path, map_location="cpu", weights_only=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(ckpt["model_name"])

    model = DINOv3Classifier(
        model_name=ckpt["model_name"],
        hidden_size=ckpt["hidden_size"],
        dropout=ckpt["dropout"],
    )

    head_state = {f"head.{k}": v for k, v in ckpt["head_state"].items()}
    model.load_state_dict(head_state, strict=False)
    model.to(device).eval()

    return model, processor, device, ckpt


# ─────────────────────────────────────────────────────────
# Load model
try:
    with st.spinner("Loading model (first time may take a minute)..."):
        model, processor, device, ckpt = load_model(HEAD_PATH)

    col1, col2, col3 = st.columns(3)
    col1.metric("Device", device.upper())
    col2.metric("Val accuracy", f"{ckpt['val_acc']:.1%}")
    col3.metric("Backbone", ckpt['model_name'].split('/')[-1])

except FileNotFoundError:
    st.error(f"❌ Model file not found: `{HEAD_PATH}`. Place `classifier_head.pth` in the same folder as `app.py`.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

st.divider()

# ─────────────────────────────────────────────────────────
# Prediction function — hard 50/50 threshold
def predict(image: Image.Image):
    pv = processor(images=image, return_tensors="pt")["pixel_values"].to(device)
    with torch.no_grad():
        probs = torch.softmax(model(pv), dim=1).cpu().numpy()[0]

    p_clean, p_dirty = float(probs[0]), float(probs[1])

    if p_dirty >= 0.5:
        label, color, emoji = "DIRTY", "#ef4444", "❌"
    else:
        label, color, emoji = "CLEAN", "#22c55e", "✅"

    return {
        "label":   label,
        "color":   color,
        "emoji":   emoji,
        "p_clean": p_clean,
        "p_dirty": p_dirty,
    }


# ─────────────────────────────────────────────────────────
# Upload
uploaded_files = st.file_uploader(
    "Upload images",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    help="Upload one or more images to classify",
)

if not uploaded_files:
    st.info("👆 Upload one or more images to start")
    st.stop()

# ─────────────────────────────────────────────────────────
# Run predictions
results = []
progress = st.progress(0, text="Predicting...")

for i, uploaded in enumerate(uploaded_files):
    image = Image.open(uploaded).convert("RGB")
    result = predict(image)
    result["filename"] = uploaded.name
    result["image"]    = image
    results.append(result)
    progress.progress(
        (i + 1) / len(uploaded_files),
        text=f"Predicting... ({i+1}/{len(uploaded_files)})",
    )

progress.empty()

# ─────────────────────────────────────────────────────────
# Summary
counts = {"CLEAN": 0, "DIRTY": 0}
for r in results:
    counts[r["label"]] += 1

st.subheader(f"📊 Results · {len(results)} image(s)")

c1, c2 = st.columns(2)
c1.metric("✅ Clean", counts["CLEAN"])
c2.metric("❌ Dirty", counts["DIRTY"])

st.divider()

# ─────────────────────────────────────────────────────────
# Display each prediction
COLS_PER_ROW = 3

for i in range(0, len(results), COLS_PER_ROW):
    cols = st.columns(COLS_PER_ROW)
    for col, r in zip(cols, results[i : i + COLS_PER_ROW]):
        with col:
            st.image(r["image"], use_container_width=True)
            st.markdown(
                f"<div style='text-align:center; padding:8px; "
                f"border:3px solid {r['color']}; border-radius:8px; "
                f"background:{r['color']}15;'>"
                f"<h3 style='color:{r['color']}; margin:0;'>"
                f"{r['emoji']} {r['label']}</h3>"
                f"<p style='margin:4px 0 0 0; font-size:13px; color:#666;'>"
                f"{r['filename']}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.progress(r["p_dirty"], text=f"Dirty: {r['p_dirty']:.1%}")
            st.progress(r["p_clean"], text=f"Clean: {r['p_clean']:.1%}")
