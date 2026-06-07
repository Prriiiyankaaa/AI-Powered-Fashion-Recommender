import streamlit as st
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import chromadb
from pathlib import Path

# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Fashion Recommender",
    
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Config
# ============================================================

BERT_MODEL_DIR = "./fashion-bert"
CHROMA_DIR = "./chromadb_store"
MAX_LEN = 64
TOP_K = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WEIGHTS = {
    "occasion": 0.3,
    "formality": 0.30,
    "constraint": 0.25,
    "color_tone": 0.15,
}

# ============================================================
# Load Label Maps
# ============================================================

@st.cache_resource
def load_label_maps():
    with open(f"{BERT_MODEL_DIR}/label_maps.json", "r") as f:
        label_maps = json.load(f)
    return label_maps

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
        self.bert = BertModel.from_pretrained("bert-base-uncased")
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
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_DIR)
    model = FashionIntentModel().to(DEVICE)
    model.load_state_dict(
        torch.load(f"{BERT_MODEL_DIR}/model.pt", map_location=DEVICE)
    )
    model.eval()
    return model, tokenizer

bert_model, tokenizer = load_bert()

# ============================================================
# Extract Intent
# ============================================================

def extract_intent(prompt, model, tokenizer):
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

    occ_scores = outputs["occasion"].softmax(dim=1).cpu().tolist()[0]
    form_scores = outputs["formality"].softmax(dim=1).cpu().tolist()[0]
    con_scores = outputs["constraint"].softmax(dim=1).cpu().tolist()[0]
    col_scores = outputs["color"].softmax(dim=1).cpu().tolist()[0]

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

def query_chromadb(intent, prompt="", top_k=TOP_K * 3):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection("fashion_products")

    total = collection.count()
    if total == 0:
        return []

    item_type = detect_item_type(prompt)
    if item_type:
        query_text = f"{item_type} {intent['occasion']} {intent['formality']}"
    else:
        query_text = f"{intent['occasion']} {intent['formality']} {intent['constraint']} outfit"

    results = collection.query(
        query_texts=[query_text],
        n_results=min(top_k, total),
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

OCCASION_LABEL_MAP = {
    "casual": "casual everyday wear",
    "party": "party outfit",
    "wedding": "wedding guest outfit",
    "office": "office wear",
    "date": "date night outfit",
}

FORMALITY_LABEL_MAP = {
    "formal": "very formal outfit",
    "semi-formal": "semi formal outfit",
    "casual": "casual informal outfit",
}

BOLDNESS_LABEL_MAP = {
    "understated": "subtle and understated outfit",
    "bold": "bold and loud statement outfit",
    "no_constraint": None,
}

COLOR_LABEL_MAP = {
    "soft": "soft muted pastel colors",
    "bright": "bright bold colors",
    "dark": "dark moody colors",
    "neutral": "neutral minimal colors",
}

def score_product(product, intent):
    score = 0.0
    score_details = {}

    # Occasion
    occ_label = OCCASION_LABEL_MAP.get(intent["occasion"], "")
    occ_match = product["occasion_scores"].get(occ_label, 0.0)
    occ_contrib = occ_match * WEIGHTS["occasion"]
    score += occ_contrib
    score_details["occasion"] = round(occ_contrib, 3)

    # Formality
    form_label = FORMALITY_LABEL_MAP.get(intent["formality"], "")
    form_match = product["formality_scores"].get(form_label, 0.0)
    form_contrib = form_match * WEIGHTS["formality"]
    score += form_contrib
    score_details["formality"] = round(form_contrib, 3)

    # Constraint
    con_label = BOLDNESS_LABEL_MAP.get(intent["constraint"])
    if con_label is None:
        con_match = 0.5
    else:
        con_match = product["boldness_scores"].get(con_label, 0.0)
    con_contrib = con_match * WEIGHTS["constraint"]
    score += con_contrib
    score_details["constraint"] = round(con_contrib, 3)

    # Color
    col_label = COLOR_LABEL_MAP.get(intent["color_tone"], "")
    col_match = product["color_scores"].get(col_label, 0.0)
    col_contrib = col_match * WEIGHTS["color_tone"]
    score += col_contrib
    score_details["color_tone"] = round(col_contrib, 3)

    return round(score, 4), score_details

def rank_products(products, intent, top_k=TOP_K):
    scored = []
    for product in products:
        score, details = score_product(product, intent)
        scored.append({**product, "match_score": score, "score_details": details})
    scored.sort(key=lambda x: x["match_score"], reverse=True)
    return scored[:top_k]

# ============================================================
# Recommend Function
# ============================================================

def recommend(prompt, top_k=TOP_K):
    intent = extract_intent(prompt, bert_model, tokenizer)
    candidates = query_chromadb(intent, prompt, top_k=top_k * 3)
    if not candidates:
        return None, None

    candidates = filter_by_item_type(prompt, candidates)
    ranked = rank_products(candidates, intent, top_k=top_k)
    
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
            intent, ranked_products = recommend(user_prompt, top_k=top_k)

        if intent is None:
            st.error("No products found. Please try a different prompt.")
        else:
            # Display Intent Analysis
            st.markdown("### Your Style Profile")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Occasion", intent["occasion"].title())
            with col2:
                st.metric("Formality", intent["formality"].title())
            with col3:
                st.metric("Constraint", intent["constraint"].title())
            with col4:
                st.metric("Color Tone", intent["color_tone"].title())

            st.divider()

            # Display Recommendations
            st.markdown("### Top Recommendations")
            
            for idx, product in enumerate(ranked_products, 1):
                with st.container(border=True):
                    col1, col2 = st.columns([2, 3])
                    
                    with col1:
                        # Display image if available
                        if product["local_image_path"] and os.path.exists(product["local_image_path"]):
                            st.image(product["local_image_path"], use_column_width=None)
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
                        st.progress(product["match_score"])
                        st.markdown(f"`{product['match_score']:.1%}`", unsafe_allow_html=True)
                        
                        # Detailed scores
                        st.markdown("**Why this match?**")
                        score_df = pd.DataFrame(
                            {
                                "Attribute": ["Occasion", "Formality", "Constraint", "Color"],
                                "Score": [
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
