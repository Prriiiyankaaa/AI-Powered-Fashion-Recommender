# Deploying to Streamlit Community Cloud

## What changed in the code

| File | Change |
|------|--------|
| `streamlit_app.py` | BERT encoder now built from `BertConfig()` instead of downloading `bert-base-uncased` at boot (weights already live in `model.pt`). ChromaDB client is cached and telemetry disabled. `use_column_width` → `use_container_width`. Progress bar value clamped to 0–1. Startup guards for missing model/DB. Optional `MODEL_URL` secret to fetch `model.pt` at runtime instead of Git LFS. `strict=False` state-dict load for cross-version safety. |
| `requirements.txt` | Rewrote with pinned versions, CPU-only PyTorch wheel index, dropped packages the app doesn't import (scikit-learn, matplotlib, seaborn, accelerate, openai-clip). |
| `runtime.txt` | New — pins Python to 3.13. |
| `.streamlit/config.toml` | New — headless server, usage stats off. |

Everything still runs locally the same way:

```bash
streamlit run streamlit_app.py
```

## Steps you need to do

### 1. Commit and push everything

```bash
git add streamlit_app.py requirements.txt runtime.txt .streamlit/config.toml DEPLOYMENT.md
git commit -m "Prepare app for Streamlit Community Cloud"
git push origin deploy
```

The large files (`fashion-bert/model.pt`, `chromadb_store/*.bin`) are already tracked with
Git LFS and are needed at runtime — leave them in the repo. Confirm they are actually
uploaded to the LFS remote:

```bash
git lfs push origin deploy --all
```

### 2. Deploy on share.streamlit.io

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. **New app** → **Deploy from an existing repo**.
3. Repository: `Prriiiyankaaa/AI-Powered-Fashion-Recommender`
   Branch: `deploy` (or merge to `main` first and use that)
   Main file path: `streamlit_app.py`
4. **Advanced settings** → set **Python version** to **3.13**.
5. Click **Deploy**. First build takes ~5–10 min (PyTorch is large).
   The first recommendation click is also slow (~30–60 s) — ChromaDB downloads its
   embedding model on first query. It's fast afterwards.

### 3. Watch out for the Git LFS quota (important)

GitHub's free tier gives **1 GB of LFS bandwidth per month**. `model.pt` is ~440 MB, and
Streamlit re-pulls it every time the container rebuilds (redeploy, dependency change, or
the periodic reboot). After ~2 rebuilds the LFS download starts returning errors and the
app breaks with a "git-lfs pointer" / checkpoint-invalid error.

Pick one:

- **Cheapest fix (recommended):** upload `model.pt` to a Hugging Face model repo (free,
  no bandwidth cap for public files), then in the Streamlit app's
  **Settings → Secrets** add:

  ```toml
  MODEL_URL = "https://huggingface.co/<your-user>/<repo>/resolve/main/model.pt"
  ```

  The app already downloads and caches it on boot when this secret is set, so you can
  also stop tracking `model.pt` in Git afterwards.
- **Or** buy a GitHub LFS data pack ($5/mo).
- **Or** just accept it for a short-lived demo and redeploy manually when it breaks.

## Resource notes

- Community Cloud gives ~2.7 GB RAM. Current load (CPU PyTorch + one BERT-base + ChromaDB
  MiniLM) sits comfortably under that.
- If you ever hit an out-of-memory reboot loop, the usual culprit is a second copy of the
  weights — the config-based load already avoids the `bert-base-uncased` download.
