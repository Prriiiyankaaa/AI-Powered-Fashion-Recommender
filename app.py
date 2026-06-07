import streamlit as st
import json
import numpy as np
import torch
import torch.nn as nn
import chromadb
import google.generativeai as genai
from transformers import BertModel, BertTokenizer
from PIL import Image
import os

# ============================================================
# CONFIG
# ============================================================

BERT_MODEL_DIR = "./fashion-bert"
CHROMA_DIR     = "./chromadb_store"
MAX_LEN        = 64
TOP_K          = 5
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

WEIGHTS = {
    "occasion":   0.35,
    "formality":  0.30,
    "constraint": 0.25,
    "color_tone": 0.15,
}

OCCASION_LABEL_MAP = {
    "casual":  "casual everyday wear",
    "party":   "party outfit",
    "wedding": "wedding guest outfit",
    "office":  "office wear",
    "date":    "date night outfit",
}
FORMALITY_LABEL_MAP = {
    "formal":      "very formal outfit",
    "semi-formal": "semi formal outfit",
    "casual":      "casual informal outfit",
}
BOLDNESS_LABEL_MAP = {
    "understated":   "subtle and understated outfit",
    "bold":          "bold and loud statement outfit",
    "no_constraint": None,
}
COLOR_LABEL_MAP = {
    "soft":    "soft muted pastel colors",
    "bright":  "bright bold colors",
    "dark":    "dark moody colors",
    "neutral": "neutral minimal colors",
}

ITEM_TITLE_KEYWORDS = {
    "jeans":    ["jeans", "denim", "bootcut"],
    "tshirt":   ["t-shirt", "tshirt", "tee", "graphic", "crew neck"],
    "shirt":    ["shirt"],
    "dress":    ["dress", "maxi", "mini", "bodycon", "gown"],
    "top":      ["top", "blouse"],
    "coord":    ["co-ord", "coord", "set"],
    "jumpsuit": ["jumpsuit", "playsuit"],
    "skirt":    ["skirt"],
    "pants":    ["pants", "trousers", "trackpants", "joggers"],
    "shorts":   ["shorts"],
}

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="StyleSense",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500&display=swap');

* { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0a;
    color: #f0ece4;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: #0a0a0a;
}

[data-testid="stHeader"] { background: transparent; }

.main-title {
    font-family: 'Playfair Display', serif;
    font-size: 4.5rem;
    font-weight: 700;
    letter-spacing: -2px;
    line-height: 1;
    background: linear-gradient(135deg, #f0ece4 0%, #c9a96e 50%, #f0ece4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}

.subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    color: #666;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 3rem;
}

.search-container {
    background: #141414;
    border: 1px solid #2a2a2a;
    border-radius: 2px;
    padding: 2rem;
    margin-bottom: 1rem;
}

.stTextArea textarea {
    background: #0a0a0a !important;
    border: 1px solid #333 !important;
    border-radius: 2px !important;
    color: #f0ece4 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    padding: 1rem !important;
    resize: none !important;
}

.stTextArea textarea:focus {
    border-color: #c9a96e !important;
    box-shadow: none !important;
}

.stTextArea textarea::placeholder { color: #444 !important; }

.stButton button {
    background: #c9a96e !important;
    color: #0a0a0a !important;
    border: none !important;
    border-radius: 2px !important;
    padding: 0.75rem 2.5rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    cursor: pointer !important;
    width: 100% !important;
    transition: all 0.2s !important;
}

.stButton button:hover {
    background: #f0ece4 !important;
    transform: translateY(-1px) !important;
}

.intent-bar {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    margin: 1.5rem 0;
}

.intent-tag {
    background: #141414;
    border: 1px solid #2a2a2a;
    border-radius: 2px;
    padding: 0.4rem 1rem;
    font-size: 0.75rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #888;
}

.intent-tag span {
    color: #c9a96e;
    margin-left: 0.4rem;
}

.reasoning-text {
    font-family: 'Playfair Display', serif;
    font-style: italic;
    color: #666;
    font-size: 0.9rem;
    margin-bottom: 2rem;
    padding-left: 1rem;
    border-left: 2px solid #2a2a2a;
}

.results-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    color: #f0ece4;
    margin-bottom: 1.5rem;
    letter-spacing: -0.5px;
}

.product-card {
    background: #141414;
    border: 1px solid #1e1e1e;
    border-radius: 2px;
    overflow: hidden;
    transition: all 0.3s ease;
    height: 100%;
}

.product-card:hover {
    border-color: #c9a96e;
    transform: translateY(-3px);
}

.product-image-container {
    width: 100%;
    aspect-ratio: 3/4;
    background: #1a1a1a;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
}

.product-image-container img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.no-image {
    font-size: 2rem;
    color: #333;
}

.product-info {
    padding: 1rem;
}

.product-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    font-weight: 500;
    color: #f0ece4;
    margin-bottom: 0.4rem;
    line-height: 1.3;
}

.product-price {
    font-family: 'Playfair Display', serif;
    font-size: 1rem;
    color: #c9a96e;
    margin-bottom: 0.75rem;
}

.match-score-bar {
    height: 2px;
    background: #1e1e1e;
    border-radius: 1px;
    margin-bottom: 0.5rem;
    overflow: hidden;
}

.match-score-fill {
    height: 100%;
    background: linear-gradient(90deg, #c9a96e, #f0ece4);
    border-radius: 1px;
}

.match-label {
    font-size: 0.7rem;
    color: #555;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.score-breakdown {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.3rem;
    margin-top: 0.75rem;
}

.score-item {
    font-size: 0.65rem;
    color: #555;
    letter-spacing: 0.5px;
}

.score-item span {
    color: #888;
}

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #2a2a2a, transparent);
    margin: 2rem 0;
}

.example-prompts {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-top: 1rem;
}

.example-chip {
    background: #141414;
    border: 1px solid #222;
    border-radius: 20px;
    padding: 0.35rem 0.85rem;
    font-size: 0.75rem;
    color: #666;
    cursor: pointer;
}

.fallback-badge {
    display: inline-block;
    background: #1a1a1a;
    border: 1px solid #333;
    border-radius: 2px;
    padding: 0.2rem 0.6rem;
    font-size: 0.65rem;
    color: #666;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-left: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODELS (cached)
# ============================================================

@st.cache_resource
def load_label_maps():
    with open(f"{BERT_MODEL_DIR}/label_maps.json", "r") as f:
        label_maps = json.load(f)
    reverse_maps = {
        cat: {int(v): k for k, v in m.items()}
        for cat, m in label_maps.items()
    }
    return label_maps, reverse_maps

@st.cache_resource
def load_gemini():
    api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")

@st.cache_resource
def load_bert_model(_label_maps):
    class FashionIntentModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.bert        = BertModel.from_pretrained("bert-base-uncased")
            hidden           = self.bert.config.hidden_size
            self.occasion_head   = nn.Linear(hidden, len(_label_maps["occasion"]))
            self.formality_head  = nn.Linear(hidden, len(_label_maps["formality"]))
            self.constraint_head = nn.Linear(hidden, len(_label_maps["constraint"]))
            self.color_head      = nn.Linear(hidden, len(_label_maps["color_tone"]))
            self.dropout         = nn.Dropout(0.3)

        def forward(self, input_ids, attention_mask):
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls = self.dropout(out.last_hidden_state[:, 0, :])
            return {
                "occasion":   self.occasion_head(cls),
                "formality":  self.formality_head(cls),
                "constraint": self.constraint_head(cls),
                "color":      self.color_head(cls),
            }

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_DIR)
    model     = FashionIntentModel().to(DEVICE)
    model.load_state_dict(torch.load(f"{BERT_MODEL_DIR}/model.pt", map_location=DEVICE))
    model.eval()
    return model, tokenizer

@st.cache_resource
def load_chromadb():
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection("fashion_products")
    return collection


# ============================================================
# INTENT EXTRACTION
# ============================================================

def extract_intent_gemini(prompt, gemini_model):
    system = f"""You are a fashion assistant for an Indian Gen Z fashion brand.
Analyze this shopping prompt and return ONLY a valid JSON object, no extra text, no markdown, no backticks.

Prompt: "{prompt}"

Return JSON with exactly these keys:
{{
  "occasion": one of [party, wedding, office, casual, date],
  "formality": one of [formal, semi-formal, casual],
  "constraint": one of [understated, bold, no_constraint],
  "color_tone": one of [soft, bright, dark, neutral],
  "item_type": one of [dress, top, shirt, jeans, pants, skirt, jumpsuit, coord, tshirt, shorts, null],
  "avoid": [],
  "reasoning": "one sentence explanation"
}}

ONLY return the JSON. Nothing else."""

    response = gemini_model.generate_content(system)
    raw      = response.text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(raw), "gemini"


def extract_intent_bert(prompt, model, tokenizer, reverse_maps):
    tokens = tokenizer(
        prompt, max_length=MAX_LEN,
        padding="max_length", truncation=True, return_tensors="pt"
    )
    input_ids      = tokens["input_ids"].to(DEVICE)
    attention_mask = tokens["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)

    occ  = outputs["occasion"].softmax(dim=1).cpu().tolist()[0]
    form = outputs["formality"].softmax(dim=1).cpu().tolist()[0]
    con  = outputs["constraint"].softmax(dim=1).cpu().tolist()[0]
    col  = outputs["color"].softmax(dim=1).cpu().tolist()[0]

    return {
        "occasion":   reverse_maps["occasion"][int(np.argmax(occ))],
        "formality":  reverse_maps["formality"][int(np.argmax(form))],
        "constraint": reverse_maps["constraint"][int(np.argmax(con))],
        "color_tone": reverse_maps["color_tone"][int(np.argmax(col))],
        "item_type":  None,
        "avoid":      [],
        "reasoning":  "Analyzed using fine-tuned BERT model",
    }, "bert"


# ============================================================
# RECOMMEND PIPELINE
# ============================================================

def query_and_rank(intent, top_k=TOP_K):
    collection = load_chromadb()
    total      = collection.count()

    item_type = intent.get("item_type")
    if item_type and item_type != "null":
        query_text = f"{item_type} {intent['occasion']} {intent['formality']}"
    else:
        query_text = f"{intent['occasion']} {intent['formality']} {intent['constraint']} outfit"

    results = collection.query(query_texts=[query_text], n_results=min(top_k * 3, total))

    products = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        try:
            products.append({
                "title":            meta.get("title", doc),
                "price":            meta.get("price", ""),
                "local_image_path": meta.get("local_image_path", ""),
                "occasion_scores":  json.loads(meta.get("occasion_scores",  "{}")),
                "boldness_scores":  json.loads(meta.get("boldness_scores",  "{}")),
                "color_scores":     json.loads(meta.get("color_scores",     "{}")),
                "formality_scores": json.loads(meta.get("formality_scores", "{}")),
            })
        except:
            continue

    # Filter by item type
    if item_type and item_type != "null":
        keywords = ITEM_TITLE_KEYWORDS.get(item_type, [])
        filtered = [p for p in products if any(kw in p["title"].lower() for kw in keywords)]
        if filtered:
            products = filtered

    # Score
    def score(product):
        s = 0.0
        d = {}
        occ_l  = OCCASION_LABEL_MAP.get(intent["occasion"], "")
        s     += (m := product["occasion_scores"].get(occ_l, 0.0)) * WEIGHTS["occasion"]
        d["occasion"] = round(m * WEIGHTS["occasion"], 3)

        form_l = FORMALITY_LABEL_MAP.get(intent["formality"], "")
        s     += (m := product["formality_scores"].get(form_l, 0.0)) * WEIGHTS["formality"]
        d["formality"] = round(m * WEIGHTS["formality"], 3)

        con_l  = BOLDNESS_LABEL_MAP.get(intent["constraint"])
        cm     = 0.5 if con_l is None else product["boldness_scores"].get(con_l, 0.0)
        s     += cm * WEIGHTS["constraint"]
        d["constraint"] = round(cm * WEIGHTS["constraint"], 3)

        col_l  = COLOR_LABEL_MAP.get(intent["color_tone"], "")
        s     += (m := product["color_scores"].get(col_l, 0.0)) * WEIGHTS["color_tone"]
        d["color_tone"] = round(m * WEIGHTS["color_tone"], 3)

        return round(s, 4), d

    scored = []
    for p in products:
        sc, det = score(p)
        scored.append({**p, "match_score": sc, "score_details": det})

    scored.sort(key=lambda x: x["match_score"], reverse=True)
    return scored[:top_k]


# ============================================================
# UI
# ============================================================

# Header
st.markdown('<div class="main-title">StyleSense</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI Fashion Recommender</div>', unsafe_allow_html=True)

# Load resources
label_maps, reverse_maps = load_label_maps()
gemini_model             = load_gemini()
bert_model, bert_tok     = load_bert_model(label_maps)

# Search box
st.markdown('<div class="search-container">', unsafe_allow_html=True)

prompt = st.text_area(
    "",
    placeholder="Tell me what you need... \"going to my friend's wedding, don't want to outshine the bride\"",
    height=100,
    label_visibility="collapsed"
)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    search_clicked = st.button("FIND MY STYLE ✦")

st.markdown('</div>', unsafe_allow_html=True)

# Example prompts
st.markdown("""
<div style="margin-top: -0.5rem; margin-bottom: 2rem;">
    <span style="font-size:0.7rem; color:#444; letter-spacing:1px; text-transform:uppercase;">Try: </span>
    <span style="font-size:0.75rem; color:#555;">my ex will be there &nbsp;·&nbsp; office but make it cute &nbsp;·&nbsp; beach vacation vibes &nbsp;·&nbsp; show me jeans</span>
</div>
""", unsafe_allow_html=True)

# Results
if search_clicked and prompt.strip():
    with st.spinner(""):

        # Extract intent
        used_fallback = False
        if gemini_model:
            try:
                intent, source = extract_intent_gemini(prompt, gemini_model)
            except Exception:
                intent, source = extract_intent_bert(prompt, bert_model, bert_tok, reverse_maps)
                used_fallback  = True
        else:
            intent, source = extract_intent_bert(prompt, bert_model, bert_tok, reverse_maps)
            used_fallback  = True

        # Reasoning
        reasoning = intent.get("reasoning", "")
        fallback_badge = '<span class="fallback-badge">BERT</span>' if used_fallback else ""
        st.markdown(f'<div class="reasoning-text">"{reasoning}" {fallback_badge}</div>', unsafe_allow_html=True)

        # Intent tags
        tag_html = '<div class="intent-bar">'
        for label, val in [("Occasion", intent["occasion"]), ("Formality", intent["formality"]),
                            ("Vibe", intent["constraint"]), ("Color", intent["color_tone"])]:
            tag_html += f'<div class="intent-tag">{label}<span>{val}</span></div>'
        if intent.get("item_type") and intent["item_type"] != "null":
            tag_html += f'<div class="intent-tag">Item<span>{intent["item_type"]}</span></div>'
        tag_html += '</div>'
        st.markdown(tag_html, unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Get results
        results = query_and_rank(intent)

        if not results:
            st.markdown('<p style="color:#555; text-align:center;">No matching products found. Try a different prompt.</p>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="results-header">Curated For You</div>', unsafe_allow_html=True)

            cols = st.columns(5)
            for i, (col, product) in enumerate(zip(cols, results)):
                with col:
                    score_pct = int(product["match_score"] * 100)

                    # Image
                    img_path = product.get("local_image_path", "")
                    if img_path and os.path.exists(img_path):
                        try:
                            img = Image.open(img_path)
                            st.image(img, use_container_width=True)
                        except:
                            st.markdown('<div style="height:200px;background:#1a1a1a;display:flex;align-items:center;justify-content:center;color:#333;font-size:2rem;">✦</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div style="height:200px;background:#1a1a1a;display:flex;align-items:center;justify-content:center;color:#333;font-size:2rem;">✦</div>', unsafe_allow_html=True)

                    # Info
                    st.markdown(f"""
                    <div class="product-info">
                        <div class="product-title">{product['title']}</div>
                        <div class="product-price">{product['price']}</div>
                        <div class="match-score-bar">
                            <div class="match-score-fill" style="width:{score_pct}%"></div>
                        </div>
                        <div class="match-label">{score_pct}% match</div>
                        <div class="score-breakdown">
                            <div class="score-item">Occasion <span>{product['score_details']['occasion']}</span></div>
                            <div class="score-item">Formality <span>{product['score_details']['formality']}</span></div>
                            <div class="score-item">Vibe <span>{product['score_details']['constraint']}</span></div>
                            <div class="score-item">Color <span>{product['score_details']['color_tone']}</span></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

elif search_clicked and not prompt.strip():
    st.markdown('<p style="color:#555; text-align:center;">Type a prompt to get started.</p>', unsafe_allow_html=True)