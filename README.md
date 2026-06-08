# AI-Powered Fashion Recommender

> *"going to my friend's birthday, want to look nice but not steal her attention"*
> → the system understands that. and recommends accordingly.

Most fashion search still works like a keyword box — you type "blue dress", you get blue dresses. This project asks: **what if the system understood what you actually meant?**

Built with a fine-tuned BERT model for intent extraction and CLIP-based visual scoring, this recommender maps natural language fashion prompts to ranked outfit recommendations — considering occasion, formality, styling constraints, and color tone simultaneously.

---

## How It Works

A user types a prompt in plain English. The system runs it through a 4-head BERT classifier that predicts **intent across four dimensions** — not just what item they want, but the social context they're dressing for. Those predicted labels are then used to score a product catalog that's been pre-embedded with CLIP, and the top matches are surfaced with explainable per-dimension scores.

```
User Prompt
    │
    ▼
FashionIntentModel (fine-tuned BERT)
    ├── occasion_head   → casual / party / wedding / office / date
    ├── formality_head  → formal / semi-formal / casual
    ├── constraint_head → understated / bold / no_constraint
    └── color_head      → soft / bright / dark / neutral
    │
    ▼
ChromaDB Vector Search  ←── product catalog embedded via CLIP
    │
    ▼
Weighted Scoring & Ranking
    │
    ▼
Top-K Recommendations (with score breakdown)
```

---

## Key Design Decisions

**Why BERT for intent, not just keyword matching?**
Prompts like *"job interview tomorrow"* carry rich contextual signals that keyword search would miss entirely. BERT encodes the full sentence and outputs structured labels — making the understanding layer model-driven rather than rule-based.

**Why CLIP for product scoring?**
CLIP embeds product images and text in the same space, allowing semantic alignment between visual product attributes (boldness, color palette, silhouette) and the predicted intent labels. Each product is pre-scored against multiple label strings (e.g., `"party outfit"`, `"subtle and understated outfit"`), making retrieval fast and explainable.

**Why ChromaDB?**
Vector similarity search lets the system retrieve candidates that are semantically close to the predicted intent — rather than relying on exact category filters. ChromaDB is lightweight, persistent, and runs locally without a separate server.

**Weighted multi-dimensional scoring**
Rather than a single similarity score, each product is scored across all four intent dimensions with tuned weights:

| Dimension  | Weight |
|------------|--------|
| Occasion   | 0.30   |
| Formality  | 0.30   |
| Constraint | 0.25   |
| Color Tone | 0.15   |

This means a product can rank highly even if it doesn't perfectly nail color, as long as it nails occasion and formality — which mirrors how humans actually prioritize outfit decisions.

---

## Tech Stack

| Component | Technology |
|---|---|
| Intent Classification | BERT (`bert-base-uncased`) fine-tuned with 4 multi-class heads |
| Visual-Semantic Scoring | CLIP (`openai-clip`) |
| Vector Database | ChromaDB (persistent, local) |
| UI | Streamlit |
| Core ML | PyTorch, HuggingFace Transformers |
| Data | pandas, scikit-learn, numpy |

---

## Project Structure

```
AI-Powered-Fashion-Recommender/
├── recommender.py        # Core pipeline: intent extraction → retrieval → scoring
├── streamlit_app.py      # Interactive web UI
├── app.py                # App entry point
├── index.html            # Static frontend (optional)
├── requirements.txt      # Dependencies
└── .streamlit/           # Streamlit config
```

---

## Setup & Run

**1. Clone the repo**
```bash
git clone https://github.com/Prriiiyankaaa/AI-Powered-Fashion-Recommender.git
cd AI-Powered-Fashion-Recommender
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add the model files**

Place your fine-tuned BERT model in `./fashion-bert/`:
```
fashion-bert/
├── model.pt
├── config.json
├── vocab.txt
└── label_maps.json
```

**4. Populate ChromaDB**

Run the catalog pipeline to embed your product catalog with CLIP and store it in ChromaDB:
```bash
python catalog_pipeline.py
```

**5. Launch the app**
```bash
streamlit run streamlit_app.py
```

Or test the recommender directly in the terminal:
```bash
python recommender.py
```

---

## Example Prompts

The system handles a wide variety of natural language inputs:

```
"going to my friend's birthday, want to look nice but not steal her attention"
→ occasion: party | formality: semi-formal | constraint: understated | color: soft

"job interview tomorrow, need to look professional"
→ occasion: office | formality: formal | constraint: understated | color: neutral

"beach vacation with my girls, something fun and colorful"
→ occasion: casual | formality: casual | constraint: bold | color: bright

"show me jeans"
→ item type filter: jeans | scored against all intent dimensions
```

---

## What's Next

- Training on a larger, more diverse prompt dataset for better intent generalization
- Adding a feedback loop so the model learns from user interactions over time
- Expanding to outfit combination recommendations (top + bottom pairing)
- Integrating real-time catalog scraping so the product index stays current

---

## About

Built by [Priyanka Agarwal](https://github.com/Prriiiyankaaa) — B.Tech CSE student at Manipal University Jaipur, interested in applied NLP and multimodal ML.
