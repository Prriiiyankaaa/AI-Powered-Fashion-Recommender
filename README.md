# AI-Powered Fashion Recommender

> *"birthday, party style but a little casual"*
> → the system understands the blend, not just the loudest word in it.

Most fashion search still works like a keyword box — you type "blue dress", you get blue dresses. This project asks: **what if the system understood what you actually meant, including the "but"?**

Built with a fine-tuned BERT model for intent extraction and a custom target-profile ranking algorithm, this recommender maps natural language style prompts to ranked outfit recommendations — reading occasion, formality, styling boldness, and colour tone as a *blend*, not a single label, and ranking products by how closely their own profile matches that blend.

## Demo

| Style Input | Style Profile |
|---|---|
| App UI | <img width="1440" height="900" alt="App UI" src="https://github.com/user-attachments/assets/08da30ba-21c3-4b67-bbe2-51fa8a2b11ab" /> | Recommendations | <img width="1440" height="900" alt="Recommendations" src="https://github.com/user-attachments/assets/31bfb080-76cf-4cc7-a184-bdd8eb01546c" /> |

## How it works

```mermaid
flowchart TD
    A["User prompt<br/><i>'birthday, party style but a little casual'</i>"] --> B["Fine-tuned BERT<br/>(intent extraction)"]
    B --> C["4 probability distributions<br/>occasion · formality · boldness · colour"]
    C --> D["Query builder<br/>(prompt text + weighted top labels)"]
    D --> E["ChromaDB semantic search<br/>(top ~60 candidates)"]
    E --> F["Target-profile ranking<br/>(cosine similarity + MMR diversity)"]
    F --> G["Top-K recommendations<br/>+ per-attribute match breakdown"]
```

A user types a prompt in plain English. A 4-head BERT classifier predicts **intent across four dimensions** — not just what item they want, but the social context they're dressing for — as a full probability distribution over each axis, not a single winning label. Those distributions drive both a semantic retrieval query and a ranking pass over the tagged product catalog, and the top matches are surfaced with explainable per-dimension scores.

### 1. Intent extraction — `streamlit_app.py`, `recommender_core.py`

| Axis | Classes |
|---|---|
| Occasion | casual · date · office · party · vacation · wedding |
| Formality | casual · semi-formal · formal |
| Constraint (boldness) | bold · no-constraint · understated |
| Colour tone | bright · dark · neutral · soft |

The logits are divided by a **temperature** (default `2.0`) before softmax, so a secondary intent ("…but a little casual") stays visible instead of being crushed by an over-confident model.

### 2. Retrieval — ChromaDB
The intent distribution is turned into a query string (the user's own words plus the top-weighted labels per axis) and embedded via ChromaDB's default sentence embedder to pull a wide candidate pool (~60 products) from the persisted vector store.

### 3. Ranking — the target-profile algorithm (`recommender_core.py`)
This is the core design decision of the project:

- Each product's tagged attribute scores are normalized into a distribution over the same label set as the intent.
- Products are ranked by **cosine similarity between the intent distribution and the product's distribution**, blended with a small raw-confidence term (`alpha`, default `0.8`).
- An optional **MMR (Maximal Marginal Relevance)** pass re-ranks the top results for diversity so recommendations aren't near-duplicates.

The practical effect: a product that is *100% "party"* now scores **lower** than one that is *70% party / 30% casual* when that's the blend the user asked for — the opposite of naive highest-score-wins ranking, and closer to how people actually prioritize outfit decisions.

### 4. The catalog — how products are tagged
Each product needs a score against every label above (e.g. "how much does this look like a *party outfit*?"). This was tuned through several iterations (see [Build history](#build-history)) and the current approach (`tag_catalog_rules.py`) is fully deterministic:

- **Garment type** is read from a code embedded in every product image URL (`BLZ` blazer, `JNS` jeans, `DRS` dress, `TRS` trousers, …) and anchors the occasion/formality distribution.
- **Colour and style keywords** in the title (solid / sequined / floral / cutout / …) refine boldness and colour tone.
- Output is written to `tagged_catalog.csv` and embedded into the `chromadb_store/` vector database the app reads from.

## Tech stack

| Component | Technology |
|---|---|
| Intent classification | BERT (`bert-base-uncased`) fine-tuned with 4 heads, KL-divergence loss against soft labels |
| Ranking | Cosine target-profile matching + MMR (custom, `recommender_core.py`) |
| Catalog tagging | Deterministic rules — garment-code + title keywords |
| Vector database | ChromaDB (persistent, local) |
| UI | Streamlit |
| Core ML | PyTorch, HuggingFace Transformers |
| Data | pandas |

## Project structure

```
streamlit_app.py         The deployed app — UI, intent extraction, orchestration
recommender_core.py      Shared ranking/retrieval logic (target-profile matching, MMR)
recommender.py           Standalone CLI harness for testing recommendations without Streamlit
tag_catalog_rules.py     Tags the catalog (garment-code + keyword rules) and builds chromadb_store/
build_dataset.py         Builds the BERT training set (adds the `vacation` class + soft/blended labels)
train.py                 Trains the intent model (KL-divergence loss against soft label distributions)
bert.ipynb               Training notebook (mirrors train.py)
label_maps_v2.json       Intent label schema
products_combined.csv    Raw scraped catalog (title, price, image URL)
tagged_catalog.csv       Catalog with computed attribute scores (output of tag_catalog_rules.py)
fashion-bert/            Trained model weights + tokenizer (Git LFS)
chromadb_store/          Persisted vector database the app queries (Git LFS)
product_images/          Product photos
requirements.txt, runtime.txt, .streamlit/config.toml   Deployment config
DEPLOYMENT.md            Streamlit Community Cloud deployment notes
```

## Setup & run

**1. Clone the repo**
```bash
git clone https://github.com/Prriiiyankaaa/AI-Powered-Fashion-Recommender.git
cd AI-Powered-Fashion-Recommender
```

**2. Install dependencies**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**3. Pull the model + vector database**

`fashion-bert/model.pt` and `chromadb_store/` are Git LFS-tracked and required to run — if they arrive as small pointer files, run:
```bash
git lfs pull
```

**4. Launch the app**
```bash
streamlit run streamlit_app.py
```

Or test the recommendation logic directly in the terminal, no UI needed:
```bash
python recommender.py
```

## Example prompts

```
"birthday, party style but a little casual"
→ occasion: party (68%) / casual (27%)  |  a genuine blend, not a coin flip

"final round interview, polished and understated"
→ occasion: office | formality: formal-leaning | constraint: understated | color: neutral

"beach vacation with my girls, something fun and colorful"
→ occasion: vacation | formality: casual | constraint: bold | color: bright

"sunday brunch with colleagues, smart-casual and put-together"
→ occasion: casual-leaning office | formality: semi-formal | constraint: relaxed
```

## Model & data

- **Intent model**: BERT-base + 4 linear heads, trained with KL-divergence against soft-label targets (`train.py`). Evaluated each epoch against a frozen, hand-written held-out set (`fashion_data_v2_heldout.csv`, 132 prompts never used for training) — `train.py` prints per-axis weighted F1 and held-out KL-divergence during training.
- **Training data**: 1,066 labelled prompts (`build_dataset.py`), including hand-written blended examples ("office party, mostly work appropriate but slightly festive" → 60% office / 40% party) so the model learns graded intent, not just single-label classification.
- **Catalog**: 296 scraped products → 262 after removing accessories/duplicates, each tagged on all 4 axes.

## Deployment

Deployed on Streamlit Community Cloud. See [DEPLOYMENT.md](DEPLOYMENT.md) for the full runbook, including hosting the model weights on Hugging Face to avoid Git LFS bandwidth limits on rebuilds.

## Build history

1. **Baseline app** — BERT intent extraction (argmax labels) → ChromaDB keyword query → weighted-sum scoring against CLIP-tagged products.
2. **Deployment hardening** — pinned dependencies, CPU-only PyTorch, config-based model loading (no redundant pretrained-weight download), cached DB client, Streamlit Cloud setup.
3. **Target-profile ranking** — replaced argmax + highest-score-wins with softmax temperature, distribution-based intent, and cosine target-matching + MMR diversity, so blended prompts produce blended, varied results instead of one repeated bucket.
4. **Retrain with soft labels** — expanded the label schema (added `vacation`), built a dataset with explicit blended/soft-label examples, retrained the intent model with KL-divergence loss.
5. **Catalog re-tagging** — diagnosed and fixed a product/image off-by-one bug in the original scrape; evaluated CLIP and FashionCLIP zero-shot tagging (both proved unreliable — FashionCLIP alone mistagged ~35% of the catalog as "office wear"); settled on a deterministic garment-code + keyword tagger, which is faster, reproducible, and more accurate for this catalog.

## Known limitations & what's next

- The catalog (262 items, fast-fashion/casual-skewed) caps how well niche prompts (e.g. formal-wear-heavy requests) can be served — the ranking is only as good as the available inventory.
- Attribute tagging is rule-based rather than visually verified per item; it's deterministic and inspectable, but not a substitute for human curation at scale.
- The `constraint` and `color_tone` axes are inherently more subjective than `occasion`/`formality` and score lower on held-out evaluation.
- Planned: a feedback loop so ranking weights adapt from user interactions; outfit-combination recommendations (top + bottom pairing); a live catalog refresh pipeline instead of a static scrape.

## About

Built by [Priyanka Agarwal](https://github.com/Prriiiyankaaa) — B.Tech CSE student at Manipal University Jaipur, interested in applied NLP and multimodal ML.
