import streamlit as st
import os
import json
import urllib.request
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer
import chromadb
from chromadb.config import Settings
from pathlib import Path

import recommender_core as rc

# Stop ChromaDB from phoning home (noisy / occasionally fatal on cloud hosts)
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Fashion Recommender",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Config
# ============================================================

BERT_MODEL_DIR = "./fashion-bert"
CHROMA_DIR = "./chromadb_store"
MODEL_PATH = os.path.join(BERT_MODEL_DIR, "model.pt")
MAX_LEN = 64
TOP_K = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Build the BERT encoder from config instead of downloading `bert-base-uncased`
# at boot: model.pt already contains every weight, so the pretrained download
# is pure startup cost / a network dependency we don't need on Streamlit Cloud.
USE_PRETRAINED_BASE = os.environ.get("USE_PRETRAINED_BASE", "0") == "1"


def _get_secret(name: str, default: str = "") -> str:
    """Read from st.secrets if a secrets file exists, else fall back to env."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, default)


# Optional: host model.pt outside the repo (e.g. to avoid Git LFS bandwidth
# limits). Set MODEL_URL in Streamlit "Secrets" to a direct-download link and
# the app fetches it once into ./fashion-bert/ on first boot.
MODEL_URL = _get_secret("MODEL_URL", "")

# Ranking / retrieval knobs live in recommender_core; expose the defaults here.
WEIGHTS = rc.WEIGHTS

# ============================================================
# Load Label Maps
# ============================================================

@st.cache_resource
def load_label_maps():
    with open(f"{BERT_MODEL_DIR}/label_maps.json", "r") as f:
        label_maps = json.load(f)
    return label_maps


def _is_real_weights(path: str) -> bool:
    """True only if `path` is the actual checkpoint, not a missing/LFS-pointer stub."""
    if not os.path.exists(path) or os.path.getsize(path) < 1_000_000:
        return False
    with open(path, "rb") as f:
        head = f.read(64)
    return not head.startswith(b"version https://git-lfs")


def ensure_model_file():
    """Make sure ./fashion-bert/model.pt is the real checkpoint, fetching MODEL_URL if not."""
    if _is_real_weights(MODEL_PATH):
        return
    if not MODEL_URL:
        st.error(
            "Model weights `fashion-bert/model.pt` are missing or unresolved "
            "(Git LFS not pulled?).\n\n"
            "Set a `MODEL_URL` secret pointing at a direct download link, "
            "or ensure Git LFS files are available to the deployment."
        )
        st.stop()
    os.makedirs(BERT_MODEL_DIR, exist_ok=True)
    with st.spinner("Downloading model weights (first run only)…"):
        tmp = MODEL_PATH + ".part"
        req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "fashion-recommender"})
        with urllib.request.urlopen(req) as resp, open(tmp, "wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        os.replace(tmp, MODEL_PATH)
    if not _is_real_weights(MODEL_PATH):
        try:
            os.remove(MODEL_PATH)
        except OSError:
            pass
        st.error(
            "The file at MODEL_URL is not a valid checkpoint (got an HTML/error page?). "
            "Use a *direct* download link, e.g. a Hugging Face `.../resolve/main/model.pt` URL."
        )
        st.stop()


for _required in (BERT_MODEL_DIR, CHROMA_DIR):
    if not os.path.exists(_required):
        st.error(f"Required path `{_required}` is missing from the deployment.")
        st.stop()

LABEL_MAPS = load_label_maps()

REVERSE_MAPS = {
    cat: {int(v): k for k, v in m.items()}
    for cat, m in LABEL_MAPS.items()
}

# ============================================================
# Model Architecture
# ============================================================

class FashionIntentModel(nn.Module):
    def __init__(self):
        super().__init__()
        if USE_PRETRAINED_BASE:
            self.bert = BertModel.from_pretrained("bert-base-uncased")
        else:
            # BertConfig() defaults == bert-base-uncased; weights come from model.pt
            self.bert = BertModel(BertConfig())
        hidden = self.bert.config.hidden_size

        self.occasion_head = nn.Linear(hidden, len(LABEL_MAPS["occasion"]))
        self.formality_head = nn.Linear(hidden, len(LABEL_MAPS["formality"]))
        self.constraint_head = nn.Linear(hidden, len(LABEL_MAPS["constraint"]))
        self.color_head = nn.Linear(hidden, len(LABEL_MAPS["color_tone"]))
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = self.dropout(output.last_hidden_state[:, 0, :])
        return {
            "occasion": self.occasion_head(cls_token),
            "formality": self.formality_head(cls_token),
            "constraint": self.constraint_head(cls_token),
            "color": self.color_head(cls_token),
        }

# ============================================================
# Load BERT Model
# ============================================================

@st.cache_resource
def load_bert():
    ensure_model_file()
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_DIR)
    model = FashionIntentModel().to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    # strict=False tolerates harmless buffer-key drift across transformers versions
    # (e.g. embeddings.position_ids); real weights are all present in model.pt.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    real_missing = [k for k in missing if "position_ids" not in k]
    if real_missing:
        st.warning(f"Model loaded with {len(real_missing)} missing weight(s): {real_missing[:5]}")
    model.eval()
    return model, tokenizer

bert_model, tokenizer = load_bert()

# ============================================================
# Extract Intent
# ============================================================

def extract_intent(prompt, model, tokenizer, temperature=rc.DEFAULT_TEMPERATURE):
    """
    Returns argmax labels (for display) plus the full softened distribution per
    axis in `scores` — the distributions are what ranking actually uses.
    A temperature > 1 keeps secondary intents ("...but a little casual") alive
    instead of letting an over-confident softmax crush them to ~0.
    """
    tokens = tokenizer(
        prompt,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = tokens["input_ids"].to(DEVICE)
    attention_mask = tokens["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)

    t = max(float(temperature), 1e-6)
    occ_scores = (outputs["occasion"] / t).softmax(dim=1).cpu().tolist()[0]
    form_scores = (outputs["formality"] / t).softmax(dim=1).cpu().tolist()[0]
    con_scores = (outputs["constraint"] / t).softmax(dim=1).cpu().tolist()[0]
    col_scores = (outputs["color"] / t).softmax(dim=1).cpu().tolist()[0]

    return {
        "occasion": REVERSE_MAPS["occasion"][int(np.argmax(occ_scores))],
        "formality": REVERSE_MAPS["formality"][int(np.argmax(form_scores))],
        "constraint": REVERSE_MAPS["constraint"][int(np.argmax(con_scores))],
        "color_tone": REVERSE_MAPS["color_tone"][int(np.argmax(col_scores))],
        "scores": {
            "occasion": occ_scores,
            "formality": form_scores,
            "constraint": con_scores,
            "color_tone": col_scores,
        },
    }

# ============================================================
# Item Type Detection
# ============================================================

ITEM_TYPE_KEYWORDS = {
    "jeans": ["jeans", "denim"],
    "tshirt": ["t-shirt", "tshirt", "tee", "graphic tee", "crew neck"],
    "dress": ["dress", "maxi", "mini", "bodycon", "gown"],
    "top": ["top", "blouse"],
    "coord": ["co-ord", "coord"],
    "jumpsuit": ["jumpsuit", "playsuit"],
    "skirt": ["skirt"],
    "pants": ["pants", "trousers", "trackpants", "joggers", "bootcut"],
    "shorts": ["shorts"],
}

ITEM_TITLE_KEYWORDS = {
    "jeans": ["jeans", "denim", "bootcut"],
    "tshirt": ["t-shirt", "tshirt", "tee", "graphic", "crew neck"],
    "dress": ["dress", "maxi", "mini", "bodycon", "gown"],
    "top": ["top", "blouse"],
    "coord": ["co-ord", "coord", "set"],
    "jumpsuit": ["jumpsuit", "playsuit"],
    "skirt": ["skirt"],
    "pants": ["pants", "trousers", "trackpants", "joggers"],
    "shorts": ["shorts"],
    "shirt": ["shirt", "office wear", "formal"],
}

def detect_item_type(prompt):
    prompt_lower = prompt.lower()
    bottom_triggers = ["pair with", "bottoms", "bottom", "goes with", "match with"]
    if any(t in prompt_lower for t in bottom_triggers):
        for item_type, keywords in ITEM_TYPE_KEYWORDS.items():
            if item_type in ["pants", "skirt", "jeans", "shorts"]:
                if any(kw in prompt_lower for kw in keywords):
                    return item_type
        return "pants"

    for item_type, keywords in ITEM_TYPE_KEYWORDS.items():
        if any(kw in prompt_lower for kw in keywords):
            return item_type

    return None

# ============================================================
# Query ChromaDB
# ============================================================

@st.cache_resource
def get_chroma_collection():
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection("fashion_products")


def query_chromadb(intent, prompt="", n_candidates=TOP_K * 4):
    collection = get_chroma_collection()

    total = collection.count()
    if total == 0:
        return []

    query_text = rc.build_query_text(
        intent["scores"], REVERSE_MAPS, prompt, detect_item_type(prompt)
    )

    results = collection.query(
        query_texts=[query_text],
        n_results=min(n_candidates, total),
    )

    products = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        try:
            products.append(
                {
                    "title": meta.get("title", doc),
                    "price": meta.get("price", ""),
                    "category": meta.get("category", ""),
                    "local_image_path": meta.get("local_image_path", ""),
                    "url": meta.get("url", ""),
                    "occasion_scores": json.loads(meta.get("occasion_scores", "{}")),
                    "boldness_scores": json.loads(meta.get("boldness_scores", "{}")),
                    "color_scores": json.loads(meta.get("color_scores", "{}")),
                    "formality_scores": json.loads(meta.get("formality_scores", "{}")),
                }
            )
        except Exception as e:
            continue

    return products

# ============================================================
# Filter by Item Type
# ============================================================

def filter_by_item_type(prompt, candidates):
    item_type = detect_item_type(prompt)

    if not item_type:
        return candidates

    title_keywords = ITEM_TITLE_KEYWORDS[item_type]
    filtered = [p for p in candidates if any(kw in p["title"].lower() for kw in title_keywords)]

    return filtered if filtered else candidates

# ============================================================
# Score and Rank
# ============================================================

# Scoring / ranking now live in recommender_core (target-profile matching).
OCCASION_LABEL_MAP = rc.OCCASION_LABEL_MAP
FORMALITY_LABEL_MAP = rc.FORMALITY_LABEL_MAP
BOLDNESS_LABEL_MAP = rc.BOLDNESS_LABEL_MAP
COLOR_LABEL_MAP = rc.COLOR_LABEL_MAP


def rank_products(products, intent, top_k=TOP_K, alpha=rc.DEFAULT_ALPHA,
                  mmr_lambda=rc.DEFAULT_MMR):
    return rc.rank_products(
        products, intent["scores"], REVERSE_MAPS,
        top_k=top_k, alpha=alpha, mmr_lambda=mmr_lambda,
    )

# ============================================================
# Recommend Function
# ============================================================

def recommend(prompt, top_k=TOP_K, temperature=rc.DEFAULT_TEMPERATURE,
              alpha=rc.DEFAULT_ALPHA, mmr_lambda=rc.DEFAULT_MMR):
    intent = extract_intent(prompt, bert_model, tokenizer, temperature=temperature)
    candidates = query_chromadb(intent, prompt, n_candidates=max(top_k * 4, 20))
    if not candidates:
        return None, None

    candidates = filter_by_item_type(prompt, candidates)
    ranked = rank_products(candidates, intent, top_k=top_k, alpha=alpha,
                           mmr_lambda=mmr_lambda)

    return intent, ranked

# ============================================================
# Streamlit UI
# ============================================================

st.title("AI- Powered Fashion Recommender")
st.markdown("Find your perfect outfit based on your style, occasion, and preferences!")

# Sidebar
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of recommendations", 1, 10, 5)

    with st.expander("Advanced tuning"):
        temperature = st.slider(
            "Nuance", 1.0, 3.5, float(rc.DEFAULT_TEMPERATURE), 0.5,
            help="Higher = take secondary intents (\"…but a little casual\") more seriously.",
        )
        target_match = st.slider(
            "Match the exact blend", 0.0, 1.0, float(rc.DEFAULT_ALPHA), 0.05,
            help="1.0 = rank purely by closeness to your profile. "
                 "0.0 = rank by raw strength on your top label (old behaviour).",
        )
        variety = st.slider(
            "Result variety", 0.0, 0.6, round(1.0 - float(rc.DEFAULT_MMR), 2), 0.05,
            help="Higher spreads picks apart so the top results aren't near-duplicates.",
        )
    mmr_lambda = 1.0 - variety

    st.divider()
    st.info("🤖 Powered by BERT intent extraction and ChromaDB semantic search")

# Main input
st.markdown("### Tell us about your style")
user_prompt = st.text_area(
    "What kind of outfit are you looking for?",
    placeholder="e.g., Beach vacation with my girls, something fun and colorful",
    height=100,
)

# Get recommendations
if st.button("🔍 Get Recommendations", use_container_width=True, type="primary"):
    if user_prompt.strip():
        with st.spinner("Analyzing your style preferences..."):
            intent, ranked_products = recommend(
                user_prompt, top_k=top_k, temperature=temperature,
                alpha=target_match, mmr_lambda=mmr_lambda,
            )

        if intent is None:
            st.error("No products found. Please try a different prompt.")
        else:
            def _blend_caption(axis):
                dist = intent["scores"][axis]
                order = sorted(range(len(dist)), key=lambda i: dist[i], reverse=True)
                bits = [
                    f"{REVERSE_MAPS[axis][i].replace('_', ' ')} {dist[i]:.0%}"
                    for i in order[:2] if dist[i] >= 0.15
                ]
                return " · ".join(bits)

            # Display Intent Analysis
            st.markdown("### Your Style Profile")
            st.caption("Ranking matches the whole blend, not just the top label.")
            col1, col2, col3, col4 = st.columns(4)
            for c, axis, title in (
                (col1, "occasion", "Occasion"),
                (col2, "formality", "Formality"),
                (col3, "constraint", "Constraint"),
                (col4, "color_tone", "Color Tone"),
            ):
                with c:
                    st.metric(title, intent[axis].replace("_", " ").title())
                    st.caption(_blend_caption(axis))

            st.divider()

            # Display Recommendations
            st.markdown("### Top Recommendations")
            
            for idx, product in enumerate(ranked_products, 1):
                with st.container(border=True):
                    col1, col2 = st.columns([2, 3])
                    
                    with col1:
                        # Display image if available
                        if product["local_image_path"] and os.path.exists(product["local_image_path"]):
                            st.image(product["local_image_path"], use_container_width=True)
                        else:
                            st.info("No image available")
                    
                    with col2:
                        # Ranking badge with medal emoji
                        rank_emoji = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else "✨"
                        st.markdown(f"### {rank_emoji} Rank #{idx}")
                        
                        # Product title with link
                        if product["url"]:
                            st.markdown(f"**[{product['title']}]({product['url']})**", unsafe_allow_html=True)
                        else:
                            st.markdown(f"**{product['title']}**")
                        
                        st.markdown(f"**Price:** {product['price']}")
                        st.markdown(f"**Category:** {product['category']}")
                        
                        # Match score with progress bar
                        st.markdown("**Match Score**")
                        st.progress(min(max(product["match_score"], 0.0), 1.0))
                        st.markdown(f"`{product['match_score']:.1%}`", unsafe_allow_html=True)
                        
                        # Per-axis closeness to the requested profile (0-1)
                        st.markdown("**Profile match by attribute**")
                        score_df = pd.DataFrame(
                            {
                                "Attribute": ["Occasion", "Formality", "Constraint", "Color"],
                                "Closeness": [
                                    product["score_details"]["occasion"],
                                    product["score_details"]["formality"],
                                    product["score_details"]["constraint"],
                                    product["score_details"]["color_tone"],
                                ],
                            }
                        )
                        st.bar_chart(score_df.set_index("Attribute"))
    else:
        st.warning("Please enter a description of the outfit you're looking for.")

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center'>
        <small>Fashion Recommender | Built with Streamlit, BERT, and ChromaDB</small>
    </div>
    """,
    unsafe_allow_html=True,
)
