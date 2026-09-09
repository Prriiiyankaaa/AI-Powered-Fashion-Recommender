# Deploying to Streamlit Community Cloud

## What's in the code now

| File | Purpose |
|------|---------|
| `streamlit_app.py` | The deployed app. BERT built from `BertConfig()` (no `bert-base-uncased` download); cached + telemetry-free ChromaDB client; `MODEL_URL` runtime download hook; startup guards; `strict=False` load. Now also: temperature on the intent softmax, target-profile ranking, tuning sliders. |
| `recommender_core.py` | **New.** Shared ranking logic (target-distance scoring, MMR, query-text builder) imported by both `streamlit_app.py` and `recommender.py` so they can't drift. |
| `recommender.py` | Local test script — now delegates scoring/ranking to `recommender_core`. |
| `requirements.txt` | Pinned versions, CPU-only PyTorch wheel index, unused packages dropped. |
| `runtime.txt` | Pins Python 3.13. |
| `.streamlit/config.toml` | Headless server, usage stats off. |

Run locally exactly as before:

```bash
streamlit run streamlit_app.py
```

---

## Phase 0 — Host the model on Hugging Face (do this, ~10 min)

`fashion-bert/model.pt` is ~440 MB. On Git LFS, GitHub's free tier only allows **1 GB
bandwidth/month**, and Streamlit re-pulls the file on every container rebuild — so after a
couple of rebuilds the app breaks with a "git-lfs pointer / invalid checkpoint" error.
Hosting the weights on Hugging Face (no bandwidth cap for public files) fixes that
permanently. The app already has the download-and-cache hook; it just needs the URL.

**Steps (yours):**

1. Create a free account at <https://huggingface.co> if you don't have one.
2. **New Model** → name it e.g. `fashion-intent-bert` → Create.
3. On the model page: **Files and versions → Add file → Upload files** → upload
   `fashion-bert/model.pt` → Commit.
   *(CLI alternative: `pip install huggingface_hub`, `huggingface-cli login`, then
   `huggingface-cli upload <you>/fashion-intent-bert fashion-bert/model.pt model.pt`.)*
4. Copy the **direct** download URL — it must contain `/resolve/main/`:
   `https://huggingface.co/<you>/fashion-intent-bert/resolve/main/model.pt`
5. In your Streamlit app dashboard → **Settings → Secrets**, add:
   ```toml
   MODEL_URL = "https://huggingface.co/<you>/fashion-intent-bert/resolve/main/model.pt"
   ```
   Save. Streamlit reboots the app; on boot it downloads and caches the weights.
6. *(Optional, keeps the repo small)* once the secret works, stop tracking the weights:
   ```bash
   git rm --cached fashion-bert/model.pt
   echo "fashion-bert/model.pt" >> .gitignore
   git commit -m "Stop tracking model.pt; served from MODEL_URL"
   ```

Nothing to change in the code for this.

---

## Phase 1 — Better recommendations (already in the code)

**Problem it fixes:** every prompt that landed on the same top labels returned the same
products, and an item that was "100% party" always beat one that was "70% party / 30%
casual" even when the blend was what you asked for.

**What changed**

- `extract_intent` divides the logits by a **temperature** (default 2.0) before softmax,
  so a secondary intent ("…but a little casual") survives instead of being crushed to ~0.
- Ranking is now **closeness to the target profile**, not highest score:
  each product's CLIP scores become a distribution, and we rank by cosine similarity to
  the intent distribution (blended with a small raw-strength term, weight `1 − alpha`).
- The ChromaDB query text now includes **your actual words** plus the weighted top-2
  labels per axis — so "beach", "colorful", "rooftop" influence retrieval.
- Optional **MMR** pass so the top results aren't near-duplicates.
- Sidebar → **Advanced tuning**: Nuance (temperature), Match the exact blend (alpha),
  Result variety (MMR). Defaults are fine; the sliders are for experimentation.
- The Style Profile now shows the blend (e.g. `party 63% · casual 29%`), and the
  per-item chart shows per-axis closeness (0–1).

Tunable via env vars too: `INTENT_TEMPERATURE`, `RANK_ALPHA`, `RANK_MMR`.

**Nothing for you to do** — it ships with the push below. If results still feel too
"argmax-y" it's because the model was trained with hard labels and is over-confident;
raise the Nuance slider, or do the Phase 2 retrain with soft labels.

---

## Push & (re)deploy

```bash
git add streamlit_app.py recommender.py recommender_core.py requirements.txt \
        runtime.txt .streamlit/config.toml DEPLOYMENT.md
git commit -m "Phase 1: temperature + target-profile ranking; Phase 0 hardening"
git push origin retrain
```

- **If the app isn't deployed yet:** <https://share.streamlit.io> → New app → repo
  `Prriiiyankaaa/AI-Powered-Fashion-Recommender`, branch `retrain`, main file
  `streamlit_app.py`, **Advanced settings → Python 3.13** → Deploy.
  First build ~5–10 min; first recommendation click ~30–60 s (ChromaDB downloads its
  embedding model once).
- **If it's already deployed** from another branch, either point it at `retrain` in the
  app settings, or merge `retrain` into that branch.
- With Phase 0 done, `git lfs push` is no longer on the critical path, but the LFS copy
  in the repo is a harmless fallback.

## Resource notes

- Community Cloud ≈ 2.7 GB RAM. CPU PyTorch + one BERT-base + ChromaDB MiniLM fits.
- OOM reboot loop → almost always a second copy of the weights; the config-based load
  avoids the `bert-base-uncased` download that used to cause it.
