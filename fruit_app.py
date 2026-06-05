import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as TF
from PIL import Image

# ── 페이지 설정 ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="과일 신선도 분류기",
    page_icon="🍎",
    layout="centered",
)

# ── 상수 ────────────────────────────────────────────────────────────────
LABELS_KO = [
    ("신선한 사과",   "🍎", True),
    ("신선한 바나나", "🍌", True),
    ("신선한 오렌지", "🍊", True),
    ("썩은 사과",     "🍎", False),
    ("썩은 바나나",   "🍌", False),
    ("썩은 오렌지",   "🍊", False),
]

HEAD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "fruit_classifier_head.pth")
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE     = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# ── 모델 로드 ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔄 모델 로딩 중...")
def load_model():
    vgg = vgg16(weights=VGG16_Weights.DEFAULT)
    backbone = nn.Sequential(
        vgg.features, vgg.avgpool, nn.Flatten(), vgg.classifier[0:3]
    ).to(DEVICE, dtype=DTYPE).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    head = nn.Sequential(
        nn.Linear(4096, 256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, 6)
    ).to(DEVICE, dtype=DTYPE)

    state = torch.load(HEAD_PATH, map_location=DEVICE, weights_only=True)
    state = {k: v.to(DTYPE) for k, v in state.items()}
    head.load_state_dict(state)
    head.eval()

    return nn.Sequential(backbone, head)

# ── 전처리 & 추론 ────────────────────────────────────────────────────────
_trans = transforms.Compose([
    transforms.Resize(232),
    transforms.CenterCrop(224),
    transforms.ToDtype(DTYPE, scale=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict(model, image: Image.Image):
    img = TF.pil_to_tensor(image.convert("RGB"))
    img = _trans(img).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        probs = torch.softmax(model(img), dim=1)[0].float().cpu()
    return probs

# ── 확률 바 HTML 생성 ────────────────────────────────────────────────────
def prob_bar(label_name, emoji, prob, is_pred, color, track_color):
    weight = "700" if is_pred else "400"
    text_color = "#111" if is_pred else "#666"
    pct = prob * 100
    highlight = f"box-shadow:0 0 0 2px {color}33;" if is_pred else ""
    return f"""
    <div style="margin-bottom:14px; padding:10px 14px; border-radius:8px;
                background:#fff; {highlight}">
        <div style="display:flex; justify-content:space-between;
                    align-items:center; margin-bottom:7px;">
            <span style="font-size:0.9rem; font-weight:{weight};
                         color:{text_color};">{emoji}&nbsp;{label_name}</span>
            <span style="font-size:0.88rem; font-weight:700;
                         color:{color};">{prob:.2%}</span>
        </div>
        <div style="background:{track_color}; border-radius:99px;
                    height:8px; overflow:hidden;">
            <div style="background:{color}; height:100%; width:{pct:.2f}%;
                        border-radius:99px;"></div>
        </div>
    </div>"""

# ── 헤더 ────────────────────────────────────────────────────────────────
st.title("🍎 과일 신선도 분류기")
st.caption(
    "VGG16 Transfer Learning &nbsp;·&nbsp; "
    "DGX Spark GB10 &nbsp;·&nbsp; "
    "Validation Accuracy **99.85%**"
)
st.divider()

model = load_model()

# ── 파일 업로더 ──────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "과일 이미지를 업로드하세요 (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    help="사과 · 바나나 · 오렌지의 신선 여부를 판별합니다",
)

if uploaded is None:
    st.markdown(
        "<div style='text-align:center; padding:60px 0; color:#aaa; font-size:1.1rem;'>"
        "📤 이미지를 업로드하면 신선도를 판단합니다"
        "</div>",
        unsafe_allow_html=True,
    )
else:
    image  = Image.open(uploaded)
    probs  = predict(model, image)
    idx    = int(probs.argmax())

    name, emoji, is_fresh = LABELS_KO[idx]
    conf   = probs[idx].item()
    color  = "#27ae60" if is_fresh else "#e74c3c"
    bg     = "#edfaf2" if is_fresh else "#fdecea"
    status = "✅ 신선함" if is_fresh else "⚠️ 부패됨"

    # ── ① 상단: 이미지 | 예측 결과 카드 ─────────────────────────────────
    col_img, col_res = st.columns([1, 1.3], gap="large")

    with col_img:
        st.image(image, use_container_width=True)

    with col_res:
        st.markdown(f"""
        <div style="background:{bg}; border-left:6px solid {color};
                    border-radius:10px; padding:22px 26px; height:100%;
                    box-sizing:border-box;">
            <div style="font-size:3rem; line-height:1.1;">{emoji}</div>
            <div style="font-size:1.5rem; font-weight:700;
                        color:{color}; margin-top:10px;">{name}</div>
            <div style="font-size:1rem; color:#555; margin-top:8px;">
                {status} &nbsp;|&nbsp; 신뢰도 &nbsp;<b>{conf:.2%}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── ② 하단: 전체 분류 확률 (st.columns으로 2칸 분리) ──────────────────
    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
    st.markdown("**전체 분류 확률**")

    col_f, col_r = st.columns(2, gap="medium")

    fresh_bars  = "".join(
        prob_bar(n, e, probs[i].item(), i == idx, "#27ae60", "#d5f0e0")
        for i, (n, e, f) in enumerate(LABELS_KO) if f
    )
    rotten_bars = "".join(
        prob_bar(n, e, probs[i].item(), i == idx, "#e74c3c", "#fad5d5")
        for i, (n, e, f) in enumerate(LABELS_KO) if not f
    )

    with col_f:
        st.markdown(
            f'<div style="background:#f5fbf7; border:1px solid #c8e8d8;'
            f'border-radius:12px; padding:16px 18px;">'
            f'<div style="font-weight:700; color:#27ae60; font-size:0.88rem;'
            f'margin-bottom:12px;">✅ 신선함</div>'
            f'{fresh_bars}</div>',
            unsafe_allow_html=True,
        )

    with col_r:
        st.markdown(
            f'<div style="background:#fdf5f5; border:1px solid #f0c8c8;'
            f'border-radius:12px; padding:16px 18px;">'
            f'<div style="font-weight:700; color:#e74c3c; font-size:0.88rem;'
            f'margin-bottom:12px;">⚠️ 부패됨</div>'
            f'{rotten_bars}</div>',
            unsafe_allow_html=True,
        )

# ── 푸터 ────────────────────────────────────────────────────────────────
st.divider()
st.caption(f"디바이스: **{DEVICE}** &nbsp;·&nbsp; dtype: **{DTYPE}**")
