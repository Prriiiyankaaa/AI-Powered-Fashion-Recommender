"""
Phase 2 - Step 1: build the v2 training set.

Reads  : fashion_data.csv            (791 rows, hard labels only)
Writes : fashion_data_v2.csv         (all rows, hard + soft labels, split, source)
         fashion_data_v2_heldout.csv  (just the frozen eval rows, for convenience)
         label_maps_v2.json           (adds the `vacation` occasion -> 5->6 classes)

What it does
------------
1. Schema (Step 0): occasion gains `vacation`. Other axes unchanged.
2. Relabel: a few original rows that are really "on holiday" (currently
   casual/date) become `vacation`. Conservative - blocked by any
   wedding/date/office/party cue.
3. Adds hand-written `vacation` rows + light paraphrase augmentation.
4. Adds hand-written *blended* rows ("party but a little casual") with SOFT
   labels, plus paraphrase augmentation that keeps the same soft labels.
5. Every row gets `*_soft` JSON columns:
      - original / single-intent rows  -> one-hot on the hard label
      - blended rows                    -> the graded distribution
   The hard `occasion/formality/...` columns are kept (= argmax of the soft
   dist) for stratification and backward compatibility.
6. Splits off a frozen, stratified held-out eval set drawn ONLY from
   hand-written rows (never from augmentation), so evaluation stays honest.

Run:  python build_dataset.py
"""

import json
import re
import random
from pathlib import Path

import pandas as pd

SEED = 42
HELDOUT_FRAC = 0.15
SRC_CSV = "fashion_data.csv"
OUT_CSV = "fashion_data_v2.csv"
OUT_HELDOUT = "fashion_data_v2_heldout.csv"
OUT_LABEL_MAPS = "label_maps_v2.json"

rng = random.Random(SEED)

# ------------------------------------------------------------------
# Step 0 - schema
# ------------------------------------------------------------------
LABEL_MAPS = {
    "occasion":   {"casual": 0, "date": 1, "office": 2, "party": 3, "vacation": 4, "wedding": 5},
    "formality":  {"casual": 0, "formal": 1, "semi-formal": 2},
    "constraint": {"bold": 0, "no_constraint": 1, "understated": 2},
    "color_tone": {"bright": 0, "dark": 1, "neutral": 2, "soft": 3},
}
VALID = {axis: set(m) for axis, m in LABEL_MAPS.items()}
AXES = ["occasion", "formality", "constraint", "color_tone"]


def _dist(axis, value):
    """value is a label string (-> one-hot) or a {label: weight} dict (-> normalised)."""
    if isinstance(value, str):
        assert value in VALID[axis], f"bad {axis} label: {value!r}"
        return {value: 1.0}
    total = float(sum(value.values()))
    out = {}
    for k, v in value.items():
        assert k in VALID[axis], f"bad {axis} label: {k!r}"
        out[k] = round(float(v) / total, 4)
    return out


def make_row(prompt, occasion, formality, constraint, color_tone, source):
    dists = {
        "occasion": _dist("occasion", occasion),
        "formality": _dist("formality", formality),
        "constraint": _dist("constraint", constraint),
        "color_tone": _dist("color_tone", color_tone),
    }
    row = {"prompt": " ".join(str(prompt).split()), "source": source}
    for axis in AXES:
        # hard = argmax, ties broken by label index (matches numpy/torch argmax)
        row[axis] = min(
            (lbl for lbl, w in dists[axis].items()
             if w == max(dists[axis].values())),
            key=lambda lbl: LABEL_MAPS[axis][lbl],
        )
        row[f"{axis}_soft"] = json.dumps(dists[axis], sort_keys=True)
    return row


# ------------------------------------------------------------------
# 1. load originals + relabel
# ------------------------------------------------------------------
orig = pd.read_csv(SRC_CSV)[["prompt", "occasion", "formality", "constraint", "color_tone"]]

VACATION_CUE = re.compile(
    r"\b(vacation|holiday|getaway|honeymoon|road ?trip|backpacking|sightseeing|"
    r"resort|staycation|city break|interrail|cruise|trip abroad|tourist)\b", re.I)
BLOCK_CUE = re.compile(
    r"\b(wedding|reception|mehndi|sangeet|roka|haldi|date|interview|office|work|"
    r"party|birthday|club|meeting|conference)\b", re.I)

rows = []
n_relabel = 0
for r in orig.itertuples(index=False):
    occ = r.occasion
    src = "original"
    if occ in ("casual", "date") and VACATION_CUE.search(r.prompt) and not BLOCK_CUE.search(r.prompt):
        occ, src, n_relabel = "vacation", "relabeled", n_relabel + 1
    rows.append(make_row(r.prompt, occ, r.formality, r.constraint, r.color_tone, src))

# ------------------------------------------------------------------
# 2. hand-written VACATION rows  (occasion is always "vacation")
#    tuple = (prompt, formality, constraint, color_tone)
# ------------------------------------------------------------------
VACATION_SEEDS = [
    ("packing for a week in goa, easy breezy outfits for the beach and shacks", "casual", "no_constraint", "bright"),
    ("european summer trip, put-together in photos but comfy for walking all day", "casual", "understated", "neutral"),
    ("tropical island honeymoon, romantic soft outfits for sunset dinners", "semi-formal", "understated", "soft"),
    ("backpacking through southeast asia, lightweight and practical, nothing precious", "casual", "understated", "neutral"),
    ("cruise vacation, dressier looks for formal nights plus daytime deck wear", "semi-formal", "no_constraint", "neutral"),
    ("road trip across rajasthan, comfortable cotton that handles heat and dust", "casual", "understated", "bright"),
    ("apres-ski drinks in the lodge after a day on the slopes", "casual", "bold", "dark"),
    ("girls trip to bali, instagrammable outfits, lots of colour and print", "casual", "bold", "bright"),
    ("family holiday in the hills, cozy layers for cool evenings and light walks", "casual", "understated", "neutral"),
    ("beach club day in mykonos, chic swimwear coverups and gold accessories", "casual", "bold", "bright"),
    ("weekend lake house getaway, relaxed cabincore, flannels and denim", "casual", "understated", "neutral"),
    ("safari in kenya, khaki and olive practical outfits", "casual", "understated", "neutral"),
    ("long haul flight outfit for a two week trip, comfiest possible but not sloppy", "casual", "understated", "neutral"),
    ("beach vacation with my girls, something fun and colourful", "casual", "bold", "bright"),
    ("resort wear for a maldives stay, flowy whites and pastels", "casual", "understated", "soft"),
    ("sightseeing in rome all day, cute but walkable, breathable fabrics", "casual", "no_constraint", "neutral"),
    ("poolside lounging on holiday, kaftans and oversized sunglasses", "casual", "no_constraint", "bright"),
    ("winter escape to the desert, warm days cold nights, layering pieces", "casual", "understated", "neutral"),
    ("tropical getaway, breezy linen sets in warm earthy tones", "casual", "understated", "neutral"),
    ("budget interrail trip, three outfits max, everything mixes and matches", "casual", "understated", "dark"),
    ("hiking holiday in the alps, moisture wicking layers and a solid shell", "casual", "understated", "dark"),
    ("city break in paris, effortless vacation style, striped tops and flats", "casual", "understated", "neutral"),
    ("holiday in a hot country, i burn easily so cover me up but keep it cute", "casual", "understated", "soft"),
    ("beach bar hopping on holiday in tulum, boho and golden hour ready", "casual", "bold", "bright"),
    ("relaxed coastal holiday, oversized shirts over swimwear", "casual", "no_constraint", "neutral"),
    ("amalfi coast trip, i want that bright vacation glamour", "semi-formal", "bold", "bright"),
    ("camping trip, functional and warm, do not care how it looks", "casual", "understated", "dark"),
    ("quiet solo retreat by the sea, soft neutral loungewear and linen", "casual", "understated", "soft"),
    ("long weekend in lisbon, comfy sneakers and easy dresses", "casual", "understated", "bright"),
    ("beach holiday, minimalist capsule, black and white swimwear and one linen shirt", "casual", "understated", "dark"),
    ("tropical resort, colourful maximalist prints, the louder the better", "casual", "bold", "bright"),
    ("vacation in a cold climate, chunky knits and a good coat for walking around town", "casual", "understated", "neutral"),
    ("sailing trip for a few days, non-slip shoes and wind proof layers", "casual", "understated", "dark"),
    ("wine country weekend, breezy midi dresses and a hat for the vineyards", "semi-formal", "understated", "soft"),
    ("beach day on holiday, just need a good coverup and sandals", "casual", "no_constraint", "neutral"),
    ("two weeks in japan in autumn, walkable layers, muted tones for the cities", "casual", "understated", "dark"),
    ("holiday in the greek islands, whites and blues, breezy and sun bleached", "casual", "understated", "soft"),
    ("mountain cabin trip over new year, warm cosy neutrals by the fire", "casual", "understated", "neutral"),
    ("beach resort, elegant poolside looks, monochrome and gold jewellery", "semi-formal", "understated", "neutral"),
    ("trekking holiday, all technical gear, comfort and function only", "casual", "understated", "dark"),
    ("seaside town holiday, nautical stripes and rolled up chinos", "casual", "understated", "neutral"),
    ("hot springs getaway in the snow, robe to pool plus warm outerwear", "casual", "no_constraint", "neutral"),
    ("family beach holiday, practical outfits that survive sand and snacks", "casual", "understated", "bright"),
    ("island hopping in the philippines, quick dry everything, salt proof", "casual", "no_constraint", "bright"),
    ("desert glamping in morocco, flowing neutrals and a warm layer for night", "casual", "understated", "neutral"),
    ("staycation spa weekend, soft loungewear and slippers energy", "casual", "no_constraint", "soft"),
    ("scenic train journey through switzerland, comfy layered neutrals for long days", "casual", "understated", "neutral"),
]

# ------------------------------------------------------------------
# 2b. destination-template augmentation for vacation
# ------------------------------------------------------------------
DESTS = [
    "goa", "bali", "greece", "italy", "thailand", "portugal", "mexico", "morocco",
    "vietnam", "sri lanka", "the maldives", "croatia", "spain", "turkey", "dubai",
    "kerala", "the andamans", "coorg", "the alps", "iceland", "norway", "a lake house",
    "the countryside", "a beach town", "a hill station", "a desert camp",
]
VIBES = [
    ("breezy and easy for the beach", "casual", "no_constraint", "bright"),
    ("comfy for walking around cities all day", "casual", "understated", "neutral"),
    ("soft flowy pieces for warm evenings", "semi-formal", "understated", "soft"),
    ("practical lightweight layers, nothing fancy", "casual", "understated", "dark"),
    ("bright colourful prints for photos", "casual", "bold", "bright"),
    ("elegant poolside looks with gold accessories", "semi-formal", "understated", "neutral"),
    ("cosy warm neutrals for cool nights", "casual", "understated", "neutral"),
    ("minimal black and white capsule", "casual", "understated", "dark"),
]
FRAMES = [
    "outfits for a trip to {d}, {v}",
    "what to pack for {d}, {v}",
    "{d} holiday, i want {v}",
    "going to {d} for a week, need {v}",
    "vacation in {d}, {v}",
]

VACATION_AUG = []
for i, d in enumerate(DESTS):
    for j in (0, 1):
        v_text, form, con, col = VIBES[(i + j * 3) % len(VIBES)]
        frame = FRAMES[(i + j * 2) % len(FRAMES)]
        VACATION_AUG.append((frame.format(d=d, v=v_text), form, con, col))

# ------------------------------------------------------------------
# 3. hand-written BLENDED rows (soft labels)
#    tuple = (prompt, occasion, formality, constraint, color_tone)
#    each field is a label string (one-hot) or a {label: weight} dict
# ------------------------------------------------------------------
BLEND_SEEDS = [
    ("it is a birthday, i want something party style but a little casual",
     {"party": 0.7, "casual": 0.3}, {"casual": 0.6, "semi-formal": 0.4},
     {"bold": 0.5, "no_constraint": 0.5}, "bright"),
    ("office party, mostly work appropriate but slightly festive",
     {"office": 0.6, "party": 0.4}, {"semi-formal": 0.6, "formal": 0.3, "casual": 0.1},
     {"understated": 0.6, "bold": 0.4}, {"neutral": 0.6, "bright": 0.4}),
    ("rooftop date but a bit party since friends join later",
     {"date": 0.6, "party": 0.4}, {"semi-formal": 0.7, "casual": 0.3},
     {"bold": 0.5, "understated": 0.5}, {"dark": 0.6, "bright": 0.4}),
    ("beach wedding guest, formal-ish but breezy for the sand",
     {"wedding": 0.75, "vacation": 0.25}, {"semi-formal": 0.6, "formal": 0.4},
     {"understated": 0.6, "no_constraint": 0.4}, {"soft": 0.6, "bright": 0.4}),
    ("work drinks straight after the office, professional but a little going-out",
     {"office": 0.55, "party": 0.45}, {"semi-formal": 0.6, "formal": 0.2, "casual": 0.2},
     {"understated": 0.55, "bold": 0.45}, "dark"),
    ("casual friday but there is a client lunch, smart but relaxed",
     {"office": 0.7, "casual": 0.3}, {"semi-formal": 0.6, "casual": 0.4},
     "understated", "neutral"),
    ("engagement party for close friends, dressy but fun and danceable",
     {"party": 0.6, "wedding": 0.4}, {"semi-formal": 0.7, "formal": 0.3},
     {"bold": 0.55, "no_constraint": 0.45}, "bright"),
    ("first date at a nice restaurant, elegant but not like i tried too hard",
     "date", {"semi-formal": 0.6, "casual": 0.4},
     {"understated": 0.7, "no_constraint": 0.3}, {"neutral": 0.6, "dark": 0.4}),
    ("birthday dinner then a club, needs to work for both",
     {"party": 0.65, "date": 0.35}, {"semi-formal": 0.55, "casual": 0.45},
     {"bold": 0.6, "understated": 0.4}, "dark"),
    ("holiday but there is one fancy dinner, mostly beach with one dressy look",
     {"vacation": 0.7, "date": 0.3}, {"casual": 0.6, "semi-formal": 0.4},
     {"understated": 0.6, "no_constraint": 0.4}, {"soft": 0.5, "bright": 0.5}),
    ("cousin's mehndi, festive and colourful but comfortable to sit on the floor",
     {"wedding": 0.8, "party": 0.2}, {"semi-formal": 0.6, "casual": 0.4},
     {"bold": 0.6, "no_constraint": 0.4}, "bright"),
    ("sunday brunch with the girls, cute and a little dressy but still casual",
     {"casual": 0.6, "party": 0.4}, {"casual": 0.6, "semi-formal": 0.4},
     {"no_constraint": 0.6, "bold": 0.4}, "bright"),
    ("date night but we are just walking and getting coffee, pretty but low effort",
     "date", "casual", {"understated": 0.7, "no_constraint": 0.3},
     {"neutral": 0.5, "soft": 0.5}),
    ("office but there is a photoshoot for the website, polished and a bit bold",
     "office", {"formal": 0.5, "semi-formal": 0.5}, {"bold": 0.6, "understated": 0.4},
     {"neutral": 0.5, "bright": 0.5}),
    ("new year's eve house party, glam but i still have to help host",
     "party", {"semi-formal": 0.6, "casual": 0.4}, {"bold": 0.7, "no_constraint": 0.3}, "dark"),
    ("diwali party at a relative's, traditional-ish but modern and not too flashy",
     {"party": 0.6, "wedding": 0.4}, {"semi-formal": 0.7, "formal": 0.3},
     {"understated": 0.55, "bold": 0.45}, "bright"),
    ("beach holiday with a boat party one afternoon, mostly relaxed one loud look",
     {"vacation": 0.65, "party": 0.35}, "casual", {"bold": 0.5, "no_constraint": 0.5}, "bright"),
    ("client dinner, formal but the restaurant is trendy so not stuffy",
     {"office": 0.6, "date": 0.4}, {"formal": 0.55, "semi-formal": 0.45},
     "understated", "dark"),
    ("friend's birthday picnic in the park, casual but photo-ready",
     {"casual": 0.6, "party": 0.4}, "casual", {"no_constraint": 0.6, "understated": 0.4},
     {"bright": 0.6, "soft": 0.4}),
    ("cocktail hour at a wedding, dressy and a bit bold but still guest-appropriate",
     "wedding", {"semi-formal": 0.6, "formal": 0.4}, {"bold": 0.55, "understated": 0.45},
     {"dark": 0.5, "bright": 0.5}),
    ("work conference, professional but i am presenting so a little statement",
     "office", {"formal": 0.6, "semi-formal": 0.4}, {"bold": 0.55, "understated": 0.45},
     {"neutral": 0.6, "bright": 0.4}),
    ("date at an art gallery opening, creative and elevated but comfortable",
     {"date": 0.6, "party": 0.4}, {"semi-formal": 0.7, "casual": 0.3},
     {"bold": 0.5, "understated": 0.5}, "dark"),
    ("sister's roka, intimate family function, understated festive",
     {"wedding": 0.7, "party": 0.3}, {"semi-formal": 0.65, "formal": 0.35},
     "understated", {"soft": 0.6, "bright": 0.4}),
    ("team offsite dinner, smart casual, relaxed but not scruffy",
     {"office": 0.55, "casual": 0.45}, {"semi-formal": 0.6, "casual": 0.4},
     "understated", "neutral"),
    ("house party but mostly close friends so comfy-glam",
     {"party": 0.7, "casual": 0.3}, "casual", {"bold": 0.5, "no_constraint": 0.5}, "dark"),
    ("valentine's dinner, romantic and a little dressy, soft not sexy",
     "date", {"semi-formal": 0.6, "casual": 0.4}, {"understated": 0.7, "bold": 0.3}, "soft"),
    ("farewell drinks for a colleague, office-to-bar, sentimental not wild",
     {"office": 0.5, "party": 0.5}, {"semi-formal": 0.6, "casual": 0.4},
     {"understated": 0.6, "bold": 0.4}, "neutral"),
    ("beach vacation but we have dinner reservations most nights, need range",
     {"vacation": 0.6, "date": 0.4}, {"casual": 0.55, "semi-formal": 0.45},
     {"no_constraint": 0.5, "understated": 0.5}, {"bright": 0.5, "soft": 0.5}),
    ("birthday brunch then shopping, cute comfortable and a bit festive",
     {"casual": 0.55, "party": 0.45}, "casual", {"no_constraint": 0.6, "bold": 0.4}, "bright"),
    ("work event at a museum, business formal but with personality",
     "office", {"formal": 0.6, "semi-formal": 0.4}, {"understated": 0.55, "bold": 0.45}, "dark"),
    ("casual engagement dinner at home, festive but relaxed",
     {"wedding": 0.55, "casual": 0.45}, {"casual": 0.6, "semi-formal": 0.4},
     {"no_constraint": 0.6, "understated": 0.4}, {"soft": 0.5, "bright": 0.5}),
    ("clubbing for a birthday but the theme is all black elegant",
     "party", {"semi-formal": 0.6, "casual": 0.4}, {"understated": 0.5, "bold": 0.5}, "dark"),
    ("office holiday lunch, festive knitwear territory, cosy but neat",
     {"office": 0.6, "party": 0.4}, {"semi-formal": 0.6, "casual": 0.4},
     "understated", {"bright": 0.5, "neutral": 0.5}),
    ("date at a live gig, edgy and fun but i still want to look nice",
     {"date": 0.5, "party": 0.5}, "casual", {"bold": 0.65, "understated": 0.35}, "dark"),
    ("sangeet night, high energy dance floor, bold colour and movement",
     {"wedding": 0.6, "party": 0.4}, {"semi-formal": 0.6, "formal": 0.4}, "bold", "bright"),
    ("smart casual networking breakfast, approachable but credible",
     "office", {"semi-formal": 0.65, "casual": 0.35}, "understated", "neutral"),
    ("beach club party on holiday, glam swimwear to cocktails",
     {"vacation": 0.5, "party": 0.5}, "casual", {"bold": 0.6, "no_constraint": 0.4}, "bright"),
    ("quiet birthday dinner with parents, nice but modest",
     {"date": 0.5, "party": 0.5}, {"semi-formal": 0.6, "casual": 0.4},
     "understated", {"neutral": 0.6, "soft": 0.4}),
    ("work from home but a video shoot at 3, presentable and a bit polished",
     "office", {"casual": 0.5, "semi-formal": 0.5}, "understated", "neutral"),
    ("reception after a registry wedding, celebratory but low key",
     "wedding", {"semi-formal": 0.6, "casual": 0.4}, {"no_constraint": 0.6, "understated": 0.4},
     {"soft": 0.5, "bright": 0.5}),
    ("date turned group hang, cute but not too date-y now that it is friends",
     {"date": 0.45, "casual": 0.35, "party": 0.2}, "casual",
     {"understated": 0.6, "no_constraint": 0.4}, "neutral"),
    ("festive office party but i am leaving early for a real party after",
     {"office": 0.4, "party": 0.6}, {"semi-formal": 0.6, "casual": 0.4},
     {"bold": 0.6, "understated": 0.4}, "dark"),
    ("beachside vows for a friend, relaxed wedding not black tie",
     {"wedding": 0.7, "vacation": 0.3}, {"semi-formal": 0.55, "casual": 0.45},
     "no_constraint", {"soft": 0.6, "bright": 0.4}),
    ("birthday party but daytime and outdoors, playful and sunny not night-out",
     {"party": 0.7, "casual": 0.3}, "casual", {"no_constraint": 0.55, "bold": 0.45}, "bright"),
]

# ------------------------------------------------------------------
# paraphrase augmentation (keeps labels identical)
# ------------------------------------------------------------------
PRE = ["", "help - ", "ok so ", "quick one: ", "hey, ", "need advice, ", "so ", "hi - "]
SUF = ["", " thanks", " any ideas?", " help pls", " what should i wear", " tia", " ideas welcome"]
SWAPS = [("i want", "i need"), ("a little", "slightly"), (" but ", " though "),
         ("something", "an outfit"), (" i am ", " i'm "), ("need", "want")]


def paraphrase(text, k):
    t = text
    a, b = SWAPS[k % len(SWAPS)]
    if a in t:
        t = t.replace(a, b, 1)
    return f"{PRE[(k * 3 + 1) % len(PRE)]}{t}{SUF[(k * 2 + 3) % len(SUF)]}".strip()


# assemble ----------------------------------------------------------
for p, f, c, cl in VACATION_SEEDS:
    rows.append(make_row(p, "vacation", f, c, cl, "seed_vacation"))
for p, f, c, cl in VACATION_AUG:
    rows.append(make_row(p, "vacation", f, c, cl, "aug_vacation"))
for k, (p, f, c, cl) in enumerate(VACATION_SEEDS):
    rows.append(make_row(paraphrase(p, k), "vacation", f, c, cl, "aug_vacation"))

for p, o, f, c, cl in BLEND_SEEDS:
    rows.append(make_row(p, o, f, c, cl, "seed_blend"))
for k, (p, o, f, c, cl) in enumerate(BLEND_SEEDS):
    rows.append(make_row(paraphrase(p, k), o, f, c, cl, "aug_blend"))
    rows.append(make_row(paraphrase(p, k + 3), o, f, c, cl, "aug_blend"))

df = pd.DataFrame(rows).drop_duplicates(subset=["prompt"]).reset_index(drop=True)

# ------------------------------------------------------------------
# 6. frozen held-out split - hand-written rows only, stratified by occasion
# ------------------------------------------------------------------
ELIGIBLE = {"original", "relabeled", "seed_vacation", "seed_blend"}
df["split"] = "train"
heldout_idx = []
for occ, grp in df[df.source.isin(ELIGIBLE)].groupby("occasion"):
    idx = list(grp.index)
    rng.shuffle(idx)
    heldout_idx += idx[: max(1, int(round(len(idx) * HELDOUT_FRAC)))]
df.loc[heldout_idx, "split"] = "heldout"

# ------------------------------------------------------------------
# validate + write
# ------------------------------------------------------------------
for r in df.itertuples(index=False):
    for axis in AXES:
        d = json.loads(getattr(r, f"{axis}_soft"))
        assert set(d) <= VALID[axis], (axis, d)
        assert abs(sum(d.values()) - 1.0) < 1e-6, (r.prompt, axis, d)

cols = ["prompt", *AXES, *[f"{a}_soft" for a in AXES], "split", "source"]
df[cols].to_csv(OUT_CSV, index=False)
df[df.split == "heldout"][cols].to_csv(OUT_HELDOUT, index=False)
Path(OUT_LABEL_MAPS).write_text(json.dumps(LABEL_MAPS, indent=2))

# ------------------------------------------------------------------
# summary
# ------------------------------------------------------------------
print(f"relabeled originals -> vacation : {n_relabel}")
print(f"total rows                      : {len(df)}  (was {len(orig)})")
print(f"  train / heldout               : {(df.split=='train').sum()} / {(df.split=='heldout').sum()}")
print("\nby source:")
print(df.source.value_counts().to_string())
print("\nhard occasion distribution:")
print(df.occasion.value_counts().to_string())
print("\nblended rows (soft label has >1 class on some axis):")
blended = df[[f"{a}_soft" for a in AXES]].apply(
    lambda r: any(len(json.loads(x)) > 1 for x in r), axis=1)
print(f"  {blended.sum()} rows")
print(f"\nwrote: {OUT_CSV}, {OUT_HELDOUT}, {OUT_LABEL_MAPS}")
