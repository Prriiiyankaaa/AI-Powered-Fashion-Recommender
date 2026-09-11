"""
Tag the catalog with deterministic rules: garment-type code (from the image URL)
+ title keywords. No CLIP, no torch, no Colab -- runs locally in seconds and
rebuilds chromadb_store/ with the local chromadb.

Why this and not CLIP: every image URL carries a garment code (BLZ blazer, JNS
jeans, TSH t-shirt, DRS dress, TRS trousers, CRD co-ord, BCH beach ...), which
is a hard, reliable signal for occasion / formality -- the axes CLIP and
FashionCLIP were noisiest on. Colour comes from the title colour word; boldness
from title style words. Output shape is identical to the CLIP tagger, so
recommender_core / streamlit_app are unchanged.

Run:  python tag_catalog_rules.py
"""
import csv
import json
import os
import re
import shutil

import chromadb
from chromadb.config import Settings

PC_CSV = "products_combined.csv"
IMAGE_DIR = "product_images"
CHROMA_DIR = "chromadb_store"
OUT_CSV = "tagged_catalog.csv"
COLL = "fashion_products"

ACCESSORY_KW = ["necklace", "earring", "bracelet", "ring", "chain", "pendant",
                "choker", "waist chain", "hair pin", "jewellery", "jewelry"]
ACCESSORY_CODES = {"NCK", "ERG", "BRC", "RNG", "HAC", "BCE"}
BLOCKLIST = {"Grey Halter Bodysuit"}

# label strings (must match recommender_core.*_LABEL_MAP values)
OCC = ["casual everyday wear", "party outfit", "wedding guest outfit",
       "office wear", "date night outfit", "beach vacation outfit"]
FORM = ["very formal outfit", "semi formal outfit", "casual informal outfit"]
BOLD = ["subtle and understated outfit", "bold and loud statement outfit"]
COL = ["soft muted pastel colors", "bright bold colors",
       "dark moody colors", "neutral minimal colors"]

CODE_RE = re.compile(r"NM-[A-Z]+-\d+-([A-Z]{2,4})-")

# ------------------------------------------------------------------
# garment code -> base weights for occasion / formality
# ------------------------------------------------------------------
G_OCC = {
    "BLZ": {"office wear": 3.0, "date night outfit": 1.0, "party outfit": 0.7, "wedding guest outfit": 0.5},
    "TRS": {"office wear": 2.6, "casual everyday wear": 1.3, "date night outfit": 0.8, "party outfit": 0.5},
    "SHT": {"office wear": 2.0, "casual everyday wear": 1.4, "date night outfit": 0.8, "party outfit": 0.8},
    "BLS": {"casual everyday wear": 1.6, "party outfit": 1.3, "date night outfit": 1.1, "office wear": 0.9, "wedding guest outfit": 0.5},
    "JNS": {"casual everyday wear": 3.2, "party outfit": 0.5, "date night outfit": 0.5, "office wear": 0.3},
    "TSH": {"casual everyday wear": 3.6, "party outfit": 0.3, "date night outfit": 0.3},
    "TAT": {"casual everyday wear": 2.4, "party outfit": 0.9, "date night outfit": 0.7},
    "SWT": {"casual everyday wear": 3.8},
    "SHO": {"casual everyday wear": 2.2, "beach vacation outfit": 1.4, "party outfit": 0.5},
    "SKO": {"casual everyday wear": 2.2, "beach vacation outfit": 1.2, "party outfit": 0.6},
    "CRD": {"beach vacation outfit": 1.5, "casual everyday wear": 1.3, "party outfit": 0.9, "date night outfit": 0.7},
    "BCH": {"beach vacation outfit": 3.6, "casual everyday wear": 0.6},
    "BDS": {"party outfit": 1.6, "casual everyday wear": 1.4, "date night outfit": 1.0},
    "SKT": {"party outfit": 1.3, "casual everyday wear": 1.3, "office wear": 0.9, "date night outfit": 0.9},
    "DRS": {"party outfit": 1.4, "date night outfit": 1.2, "wedding guest outfit": 1.2, "casual everyday wear": 0.7, "beach vacation outfit": 0.4},
    "JMP": {"date night outfit": 1.5, "party outfit": 1.3, "casual everyday wear": 0.9, "office wear": 0.5, "beach vacation outfit": 0.3},
}
G_FORM = {
    "BLZ": {"very formal outfit": 2.6, "semi formal outfit": 2.0, "casual informal outfit": 0.3},
    "TRS": {"very formal outfit": 1.4, "semi formal outfit": 2.0, "casual informal outfit": 1.2},
    "SHT": {"very formal outfit": 1.2, "semi formal outfit": 2.2, "casual informal outfit": 1.2},
    "BLS": {"very formal outfit": 0.7, "semi formal outfit": 1.8, "casual informal outfit": 1.8},
    "JNS": {"very formal outfit": 0.1, "semi formal outfit": 0.7, "casual informal outfit": 3.4},
    "TSH": {"very formal outfit": 0.1, "semi formal outfit": 0.4, "casual informal outfit": 3.8},
    "TAT": {"semi formal outfit": 0.8, "casual informal outfit": 2.8},
    "SWT": {"casual informal outfit": 3.8},
    "SHO": {"casual informal outfit": 3.4, "semi formal outfit": 0.5},
    "SKO": {"casual informal outfit": 3.0, "semi formal outfit": 0.7},
    "CRD": {"semi formal outfit": 1.8, "casual informal outfit": 1.8},
    "BCH": {"casual informal outfit": 3.4},
    "BDS": {"semi formal outfit": 1.4, "casual informal outfit": 1.8},
    "SKT": {"very formal outfit": 0.7, "semi formal outfit": 2.0, "casual informal outfit": 1.4},
    "DRS": {"very formal outfit": 1.0, "semi formal outfit": 2.2, "casual informal outfit": 1.2},
    "JMP": {"very formal outfit": 0.6, "semi formal outfit": 2.0, "casual informal outfit": 1.6},
}
G_DEFAULT_OCC = {"casual everyday wear": 1.6, "party outfit": 1.0, "date night outfit": 1.0,
                 "office wear": 0.9, "wedding guest outfit": 0.6, "beach vacation outfit": 0.4}
G_DEFAULT_FORM = {"very formal outfit": 0.7, "semi formal outfit": 1.6, "casual informal outfit": 1.6}

# ------------------------------------------------------------------
# title keyword cues (added on top of the garment base)
# ------------------------------------------------------------------
TITLE_OCC = [
    ("cargo", {"casual everyday wear": 3}), ("track", {"casual everyday wear": 3}),
    ("jogger", {"casual everyday wear": 3}), ("sweatpant", {"casual everyday wear": 3}),
    ("pyjama", {"casual everyday wear": 3}), ("legging", {"casual everyday wear": 2.5}),
    ("graphic", {"casual everyday wear": 2.5}), ("typograph", {"casual everyday wear": 2.5}),
    ("denim", {"casual everyday wear": 2}), ("oversized", {"casual everyday wear": 1.5}),
    ("hoodie", {"casual everyday wear": 3}), ("crew neck", {"casual everyday wear": 1.2}),
    ("pleated pant", {"office wear": 2}), ("pinstripe", {"office wear": 2.5}),
    ("tailored", {"office wear": 2}), ("formal", {"office wear": 2.5}),
    ("buttoned", {"office wear": 1.5}), ("button down", {"office wear": 1.5}),
    ("button-down", {"office wear": 1.5}), ("lapel", {"office wear": 1.5}),
    ("waistcoat", {"office wear": 2}), ("pencil", {"office wear": 1.5}),
    ("sequin", {"party outfit": 3, "date night outfit": 1.5}),
    ("sparkle", {"party outfit": 2.5}), ("glitter", {"party outfit": 2.5}),
    ("bodycon", {"party outfit": 2.5, "date night outfit": 1.5}),
    ("corset", {"party outfit": 2.5, "date night outfit": 1}),
    ("bustier", {"party outfit": 2.5}), ("backless", {"party outfit": 2, "date night outfit": 1.5}),
    ("cutout", {"party outfit": 1.8, "date night outfit": 1}),
    ("cut out", {"party outfit": 1.8, "date night outfit": 1}),
    ("sheer", {"party outfit": 1.5, "date night outfit": 1}),
    ("mesh", {"party outfit": 1.5}), ("halter", {"party outfit": 1.3, "date night outfit": 1}),
    ("one shoulder", {"party outfit": 1.3, "date night outfit": 1}),
    ("strapless", {"party outfit": 1.5}), ("slit", {"party outfit": 1, "date night outfit": 1}),
    ("rhinestone", {"party outfit": 2}),
    ("gown", {"wedding guest outfit": 3}), ("embellished", {"wedding guest outfit": 2.2, "party outfit": 1}),
    ("embroidered", {"wedding guest outfit": 2.2}), ("ethnic", {"wedding guest outfit": 2, "party outfit": 1}),
    ("anarkali", {"wedding guest outfit": 3}), ("lehenga", {"wedding guest outfit": 3}),
    ("brocade", {"wedding guest outfit": 2}), ("organza", {"wedding guest outfit": 2}),
    ("maxi", {"wedding guest outfit": 1.2, "beach vacation outfit": 0.8, "date night outfit": 0.6}),
    ("wrap dress", {"date night outfit": 2, "casual everyday wear": 1}),
    ("midi", {"date night outfit": 1, "office wear": 0.8}),
    ("satin", {"date night outfit": 1.5, "party outfit": 1}),
    ("slip dress", {"date night outfit": 2}),
    ("linen", {"beach vacation outfit": 2.5}), ("crochet", {"beach vacation outfit": 2.5}),
    ("kaftan", {"beach vacation outfit": 3}), ("resort", {"beach vacation outfit": 3}),
    ("schiffli", {"beach vacation outfit": 1.5}), ("tropical", {"beach vacation outfit": 1.5}),
    ("floral", {"beach vacation outfit": 0.6, "date night outfit": 0.4}),
    ("crop", {"casual everyday wear": 0.8, "party outfit": 0.6}),
]
TITLE_FORM = [
    ("cargo", {"casual informal outfit": 3}), ("track", {"casual informal outfit": 3}),
    ("jogger", {"casual informal outfit": 3}), ("graphic", {"casual informal outfit": 2}),
    ("oversized", {"casual informal outfit": 1.5}), ("denim", {"casual informal outfit": 1.5}),
    ("formal", {"very formal outfit": 2.5}), ("tailored", {"very formal outfit": 2}),
    ("pinstripe", {"very formal outfit": 2}), ("pleated pant", {"very formal outfit": 1.5}),
    ("buttoned", {"semi formal outfit": 1.2}), ("lapel", {"very formal outfit": 1.2}),
    ("gown", {"very formal outfit": 1.5, "semi formal outfit": 1}),
    ("embellished", {"semi formal outfit": 1}), ("sequin", {"semi formal outfit": 1}),
    ("bodycon", {"semi formal outfit": 1}), ("solid", {"semi formal outfit": 0.4}),
]
TITLE_BOLD = {
    "bold and loud statement outfit": ["sequin", "sequined", "sparkle", "glitter", "cutout",
        "cut out", "backless", "sheer", "mesh", "one shoulder", "print", "printed", "floral",
        "geometric", "ethnic", "graphic", "typograph", "embellished", "rhinestone", "ruffle",
        "ruched", "slit", "embroidered", "lace", "colourblock", "colorblock", "colour block",
        "tie-up", "corset", "polka dot", "striped", "houndstooth", "leopard", "abstract",
        "tie dye", "tie-dye", "animal", "checkered", "plaid"],
    "subtle and understated outfit": ["solid", "basic", "plain", "minimal", "classic", "clean",
        "simple", "fitted", "straight fit", "regular fit", "tailored", "monochrome"],
}
TITLE_COL = {
    "soft muted pastel colors": ["pastel", "blush", "lavender", "peach", "mint", "powder",
                                 "lilac", "baby ", "sky ", "ivory", "cream", "champagne"],
    "neutral minimal colors": ["white", "off white", "off-white", "grey", "gray", "beige",
                               "tan", "nude", "khaki", "stone", "taupe", "camel", "ecru", "sand"],
    "dark moody colors": ["black", "navy", "charcoal", "maroon", "wine", "burgundy", "forest",
                          "olive", "brown", "chocolate", "espresso", "dark blue", "dark green"],
    "bright bold colors": ["red", "pink", "orange", "yellow", "fuchsia", "magenta", "coral",
                           "lime", "neon", "gold", "mustard", "rust", "green", "cobalt",
                           "electric", "teal", "turquoise", "purple", "violet", "bright",
                           "multi color", "multicolor", "multicolour", "colourful", "colorful"],
}
FLOOR = {"occ": 0.05, "form": 0.08, "bold": 0.15, "col": 0.10}


def _norm(d, floor, labels):
    v = {l: d.get(l, 0.0) + floor for l in labels}
    s = sum(v.values())
    return {l: round(x / s, 4) for l, x in v.items()}


def _apply(base, cues, title):
    d = dict(base)
    for kw, votes in cues:
        if kw in title:
            for l, w in votes.items():
                d[l] = d.get(l, 0.0) + w
    return d


def _kw_score(title, table):
    d = {}
    for label, kws in table.items():
        d[label] = sum(1.0 for kw in kws if kw in title)
    return d


def tag_row(title, code):
    t = title.lower()
    occ = _apply(G_OCC.get(code, G_DEFAULT_OCC), TITLE_OCC, t)
    form = _apply(G_FORM.get(code, G_DEFAULT_FORM), TITLE_FORM, t)
    bold = _kw_score(t, TITLE_BOLD)
    col = _kw_score(t, TITLE_COL)
    if re.search(r"\blight [a-z]", t):
        col["soft muted pastel colors"] = col.get("soft muted pastel colors", 0) + 2
    if "blue" in t and not any(x in t for x in ("dark blue", "navy", "light blue", "sky blue", "powder blue")):
        col["bright bold colors"] = col.get("bright bold colors", 0) + 1
    return (
        _norm(occ, FLOOR["occ"], OCC),
        _norm(form, FLOOR["form"], FORM),
        _norm(bold, FLOOR["bold"], BOLD),
        _norm(col, FLOOR["col"], COL),
    )


def main():
    rows = list(csv.DictReader(open(PC_CSV, newline="")))
    print(f"{len(rows)} products in {PC_CSV}")

    out = []
    for r in rows:
        pid = int(r["product_id"])
        img = f"{IMAGE_DIR}/{pid - 1}.jpg"          # 0-indexed image fix
        if not os.path.exists(img):
            continue
        title = r["title"]
        if any(k in title.lower() for k in ACCESSORY_KW) or title in BLOCKLIST:
            continue
        m = CODE_RE.search(r.get("image_url", "") or "")
        code = m.group(1) if m else None
        if code in ACCESSORY_CODES:
            continue
        occ, form, bold, col = tag_row(title, code)
        out.append({
            "product_id": pid, "title": title, "price": r["price"],
            "image_url": r.get("image_url", ""), "url": r.get("url", ""),
            "local_image_path": img, "garment_code": code or "",
            "occasion_scores": json.dumps(occ), "boldness_scores": json.dumps(bold),
            "color_scores": json.dumps(col), "formality_scores": json.dumps(form),
        })
    print(f"tagged {len(out)} products")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"wrote {OUT_CSV}")

    # rebuild ChromaDB locally
    shutil.rmtree(CHROMA_DIR, ignore_errors=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR,
                                       settings=Settings(anonymized_telemetry=False))
    coll = client.create_collection(COLL)
    ids, docs, mets = [], [], []
    for r in out:
        occ_s = json.loads(r["occasion_scores"])
        bold_s = json.loads(r["boldness_scores"])
        ids.append(str(r["product_id"]))
        docs.append(f"{r['title']} {max(occ_s, key=occ_s.get)} {max(bold_s, key=bold_s.get)}")
        mets.append({k: r[k] for k in ("title", "price", "url", "local_image_path",
                                       "occasion_scores", "boldness_scores",
                                       "color_scores", "formality_scores")} | {"category": ""})
    for i in range(0, len(ids), 100):
        coll.add(ids=ids[i:i + 100], documents=docs[i:i + 100], metadatas=mets[i:i + 100])
    print(f"ChromaDB rebuilt: {coll.count()} products")

    # summary
    import collections as _c
    for ax in ("occasion_scores", "formality_scores", "boldness_scores", "color_scores"):
        cnt = _c.Counter()
        for r in out:
            d = json.loads(r[ax])
            cnt[max(d, key=d.get).split()[0]] += 1
        print(f"  {ax:17s} {dict(cnt)}")
    print("\n  samples:")
    for want in ["Bootcut Pants", "Lapel Collar Blazer", "Sequin", "Cargo Jeans",
                 "Crew Neck T-Shirt", "Backless Scoop", "Maxi Dress", "Linen"]:
        for r in out:
            if want.lower() in r["title"].lower():
                o = json.loads(r["occasion_scores"]); fm = json.loads(r["formality_scores"])
                print(f"    {r['title'][:42]:42s} [{r['garment_code']:3s}] "
                      f"{max(o, key=o.get).split()[0]:8s} / {max(fm, key=fm.get).split()[0]}")
                break


if __name__ == "__main__":
    main()
