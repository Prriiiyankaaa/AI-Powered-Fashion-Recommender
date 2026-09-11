# ============================================================
# Fashion Recommender — Complete Notebook
# Run cells in order from top to bottom every session
# ============================================================


# ============================================================
# CELL 1 — Imports
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import chromadb

import recommender_core as rc


# ============================================================
# CELL 2 — Config
# ============================================================

BERT_MODEL_DIR = "./fashion-bert"
CHROMA_DIR     = "./chromadb_store"
MAX_LEN        = 64
TOP_K          = 5
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

WEIGHTS = {
    "occasion":   0.3,
    "formality":  0.30,
    "constraint": 0.25,
    "color_tone": 0.15,
}

print(f"Device: {DEVICE}")
print("Config loaded.")


# ============================================================
# CELL 3 — Label Maps
# ============================================================

with open(f"{BERT_MODEL_DIR}/label_maps.json", "r") as f:
    LABEL_MAPS = json.load(f)

REVERSE_MAPS = {
    cat: {int(v): k for k, v in m.items()}
    for cat, m in LABEL_MAPS.items()
}

print("Label maps loaded:")
for cat, mapping in LABEL_MAPS.items():
    print(f"  {cat}: {mapping}")


# ============================================================
# CELL 4 — Model Architecture
# ============================================================

class FashionIntentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert    = BertModel.from_pretrained("bert-base-uncased")
        hidden       = self.bert.config.hidden_size

        self.occasion_head   = nn.Linear(hidden, len(LABEL_MAPS["occasion"]))
        self.formality_head  = nn.Linear(hidden, len(LABEL_MAPS["formality"]))
        self.constraint_head = nn.Linear(hidden, len(LABEL_MAPS["constraint"]))
        self.color_head      = nn.Linear(hidden, len(LABEL_MAPS["color_tone"]))
        self.dropout         = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask):
        output    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = self.dropout(output.last_hidden_state[:, 0, :])
        return {
            "occasion":   self.occasion_head(cls_token),
            "formality":  self.formality_head(cls_token),
            "constraint": self.constraint_head(cls_token),
            "color":      self.color_head(cls_token),
        }

print("Model class defined.")


# ============================================================
# CELL 5 — Load BERT
# ============================================================

def load_bert():
    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_DIR)

    print("Building model architecture...")
    model = FashionIntentModel().to(DEVICE)

    print("Loading trained weights...")
    model.load_state_dict(
        torch.load(f"{BERT_MODEL_DIR}/model.pt", map_location=DEVICE)
    )
    model.eval()
    print("BERT ready.")
    return model, tokenizer

bert_model, tokenizer = load_bert()


# ============================================================
# CELL 6 — Extract Intent
# ============================================================

def extract_intent(prompt, model, tokenizer, temperature=rc.DEFAULT_TEMPERATURE):
    tokens = tokenizer(
        prompt,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids      = tokens["input_ids"].to(DEVICE)
    attention_mask = tokens["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)

    t = max(float(temperature), 1e-6)
    occ_scores  = (outputs["occasion"]   / t).softmax(dim=1).cpu().tolist()[0]
    form_scores = (outputs["formality"]  / t).softmax(dim=1).cpu().tolist()[0]
    con_scores  = (outputs["constraint"] / t).softmax(dim=1).cpu().tolist()[0]
    col_scores  = (outputs["color"]      / t).softmax(dim=1).cpu().tolist()[0]

    return {
        "occasion":   REVERSE_MAPS["occasion"][int(np.argmax(occ_scores))],
        "formality":  REVERSE_MAPS["formality"][int(np.argmax(form_scores))],
        "constraint": REVERSE_MAPS["constraint"][int(np.argmax(con_scores))],
        "color_tone": REVERSE_MAPS["color_tone"][int(np.argmax(col_scores))],
        "scores": {
            "occasion":   occ_scores,
            "formality":  form_scores,
            "constraint": con_scores,
            "color_tone": col_scores,
        }
    }

# Quick test
test_intent = extract_intent("friend's birthday, want to look nice but not steal attention", bert_model, tokenizer)
print("Intent extraction working:")
print(f"  occasion={test_intent['occasion']} | formality={test_intent['formality']} | constraint={test_intent['constraint']} | color={test_intent['color_tone']}")


# ============================================================
# CELL 7 — Item Type Detection (used by both query and filter)
# ============================================================

# Maps item type name → keywords to detect in user prompt
ITEM_TYPE_KEYWORDS = {
    "jeans":    ["jeans", "denim"],
    "tshirt":   ["t-shirt", "tshirt", "tee", "graphic tee", "crew neck"],
    "dress":    ["dress", "maxi", "mini", "bodycon", "gown"],
    "top":      ["top", "blouse"],
    "coord":    ["co-ord", "coord"],
    "jumpsuit": ["jumpsuit", "playsuit"],
    "skirt":    ["skirt"],
    "pants":    ["pants", "trousers", "trackpants", "joggers", "bootcut"],
    "shorts":   ["shorts"],
}

# Maps item type name → keywords to match against product titles in ChromaDB
ITEM_TITLE_KEYWORDS = {
    "jeans":    ["jeans", "denim", "bootcut"],
    "tshirt":   ["t-shirt", "tshirt", "tee", "graphic", "crew neck"],
    "dress":    ["dress", "maxi", "mini", "bodycon", "gown"],
    "top":      ["top", "blouse"],
    "coord":    ["co-ord", "coord", "set"],
    "jumpsuit": ["jumpsuit", "playsuit"],
    "skirt":    ["skirt"],
    "pants":    ["pants", "trousers", "trackpants", "joggers"],
    "shorts":   ["shorts"],
    "shirt":    ["shirt","office wear","formal"],  
}
def detect_item_type(prompt):
    prompt_lower = prompt.lower()

    bottom_triggers = ["pair with", "bottoms", "bottom", "goes with", "match with"]
    if any(t in prompt_lower for t in bottom_triggers):
        # Suggest pants/skirts as bottoms
        for item_type, keywords in ITEM_TYPE_KEYWORDS.items():
            if item_type in ["pants", "skirt", "jeans", "shorts"]:
                if any(kw in prompt_lower for kw in keywords):
                    return item_type
        return "pants"  # default bottom suggestion

    # Normal detection
    for item_type, keywords in ITEM_TYPE_KEYWORDS.items():
        if any(kw in prompt_lower for kw in keywords):
            return item_type

    return None

print("Item type detection defined.")


# CELL 8 — Query ChromaDB

def query_chromadb(intent, prompt="", top_k=TOP_K * 3):
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection("fashion_products")

    total = collection.count()
    if total == 0:
        print("ChromaDB is empty. Run catalog_pipeline.py first.")
        return []

    # Query text = user's own words + weighted top labels (see recommender_core)
    query_text = rc.build_query_text(
        intent["scores"], REVERSE_MAPS, prompt, detect_item_type(prompt)
    )

    print(f"ChromaDB query: '{query_text}'")

    results = collection.query(
        query_texts=[query_text],
        n_results=min(top_k, total),
    )

    products = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        try:
            products.append({
                "title":            meta.get("title", doc),
                "price":            meta.get("price", ""),
                "category":         meta.get("category", ""),
                "local_image_path": meta.get("local_image_path", ""),
                "occasion_scores":  json.loads(meta.get("occasion_scores",  "{}")),
                "boldness_scores":  json.loads(meta.get("boldness_scores",  "{}")),
                "color_scores":     json.loads(meta.get("color_scores",     "{}")),
                "formality_scores": json.loads(meta.get("formality_scores", "{}")),
            })
        except Exception as e:
            print(f"Skipping product: {e}")
            continue

    print(f"Retrieved {len(products)} candidates.")
    return products

print("query_chromadb() defined.")



#  Filter by Item Type


def filter_by_item_type(prompt, candidates):
    item_type = detect_item_type(prompt)

    if not item_type:
        return candidates  # no item type in prompt — return all

    title_keywords = ITEM_TITLE_KEYWORDS[item_type]
    filtered = [
        p for p in candidates
        if any(kw in p["title"].lower() for kw in title_keywords)
    ]

    print(f"Item filter '{item_type}': {len(filtered)}/{len(candidates)} matched")

    # Fallback — if nothing matched titles, return all candidates
    return filtered if filtered else candidates

print("filter_by_item_type() defined.")



# Score and Rank — target-profile matching lives in recommender_core

OCCASION_LABEL_MAP  = rc.OCCASION_LABEL_MAP
FORMALITY_LABEL_MAP = rc.FORMALITY_LABEL_MAP
BOLDNESS_LABEL_MAP  = rc.BOLDNESS_LABEL_MAP
COLOR_LABEL_MAP     = rc.COLOR_LABEL_MAP


def score_product(product, intent):
    return rc.score_product(product, intent["scores"], REVERSE_MAPS)


def rank_products(products, intent, top_k=TOP_K):
    return rc.rank_products(products, intent["scores"], REVERSE_MAPS, top_k=top_k)

print("Scoring functions defined.")



# CELL 11 — Display Results


def display_results(prompt, intent, ranked_products):
    print("\n" + "=" * 60)
    print(f"Prompt    : {prompt}")
    print(f"Occasion  : {intent['occasion']}")
    print(f"Formality : {intent['formality']}")
    print(f"Constraint: {intent['constraint']}")
    print(f"Color     : {intent['color_tone']}")
    print("=" * 60)
    print(f"\nTop {len(ranked_products)} Recommendations:\n")
    for i, p in enumerate(ranked_products):
        print(f"{i+1}. {p['title']}")
        print(f"   Price      : {p['price']}")
        print(f"   Match Score: {p['match_score']}")
        print(f"   Why matched: occasion={p['score_details']['occasion']} | "
              f"formality={p['score_details']['formality']} | "
              f"constraint={p['score_details']['constraint']} | "
              f"color={p['score_details']['color_tone']}")
        print()

print("display_results() defined.")



OCCASION_WORDS = [
    "party", "wedding", "office", "date", "birthday", "interview",
    "casual", "vacation", "formal", "night out", "festival", "puja",
    "college", "farewell", "reunion", "brunch", "dinner", "event",
]

def recommend(prompt, top_k=TOP_K):
    # Step 1 — Extract intent
    intent = extract_intent(prompt, bert_model, tokenizer)

    # Step 2 — Query ChromaDB (wider pool so ranking/diversity has room to work)
    candidates = query_chromadb(intent, prompt, top_k=max(top_k * 10, 60))
    if not candidates:
        print("No products found.")
        return []

    # Step 3 — Filter by item type
    candidates = filter_by_item_type(prompt, candidates)

    # Step 4 — Always score, rank and display with full format
    ranked = rank_products(candidates, intent, top_k=top_k)
    display_results(prompt, intent, ranked)
    return ranked
print("recommend() defined.")
print("\nAll cells loaded. Ready to use.\n")


# ============================================================
# CELL 13 — Test Prompts
# ============================================================

test_prompts = [
    "going to my friend's birthday, want to look nice but not steal her attention",
    "job interview tomorrow, need to look professional",
    "my ex will be there, i need to look amazing",
    "show me jeans",
    "i want a t-shirt",
    "beach vacation with my girls, something fun and colorful",
]

for prompt in test_prompts:
    recommend(prompt)
    print()


# CELL 14 — Interactive Mode


print("=" * 60)
print("Interactive Mode — type a prompt, press Enter")
print("Type 'quit' to exit")
print("=" * 60 + "\n")

while True:
    user_input = input("Your prompt: ").strip()
    if user_input.lower() == "quit":
        break
    if user_input:
        recommend(user_input)
        print()