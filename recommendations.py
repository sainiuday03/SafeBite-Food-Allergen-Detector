"""
SafeBite - recommendations.py (fixed version)

Pipeline:
  OCR text -> check_allergens() -> deduce_product_category()
           -> Gemini recommends allergen-free products of the SAME type
           -> results are sanity-filtered
           -> hardcoded OFFLINE_BACKUP only if Gemini fails

What changed vs. the old file:
  * Uses client.models.generate_content (the stable Gemini call).
  * Forces JSON output, so parsing no longer breaks on stray text.
  * Tries a chain of model names and auto-discovers valid ones, so one wrong
    model name no longer silently sends everything to the hardcoded list.
  * Prompt now includes the scanned product text, so a cake gets cakes.
  * Products that still contain the user's allergen words are filtered out.
  * LAST_STATUS records whether Gemini or the backup was used (and why).
"""

import os
import re
import io
import json
import time
import hashlib
from difflib import SequenceMatcher

from google import genai
from google.genai import types

import logging
logging.getLogger("google_genai").setLevel(logging.ERROR)   # hides the harmless 'automatic function calling' notice

# --------------------------------------------------------------------------
# API KEY + CLIENT
# --------------------------------------------------------------------------
def _get_api_key():
    """Reads the key from .streamlit/secrets.toml, then environment variables."""
    try:
        import streamlit as st
        key = st.secrets.get("GEMINI_API_KEY")
        if key:
            return str(key).strip()
    except Exception:
        pass
    return (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()


_client = None

def get_client():
    global _client
    if _client is None:
        key = _get_api_key()
        if not key:
            raise RuntimeError("No GEMINI_API_KEY found in .streamlit/secrets.toml or environment.")
        _client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=20000))
    return _client


# Shown in the app so you can see what actually happened during a scan.
LAST_STATUS = {"engine": None, "model": None, "error": None, "queries": [], "sources": [], "detected": ""}

# --------------------------------------------------------------------------
# ALLERGEN KEYWORDS + ADVICE
# --------------------------------------------------------------------------
allergy_keywords = {
    "Milk / Dairy": ["milk", "dairy", "ghee", "butter", "cheese", "cream", "whey",
                     "lactose", "curd", "paneer", "milk solids", "milksolids", "casein"],
    "Nuts / Peanuts": ["nut", "peanut", "almond", "cashew", "walnut", "hazelnut",
                       "pistachio", "groundnut", "ground nut"],
    "Gluten / Wheat": ["wheat", "gluten", "barley", "rye", "maida", "suji", "atta",
                       "wheat flour", "malt", "malt extract", "semolina"],
    "Soy": ["soy", "soya", "soybean", "soyabean", "lecithin", "soya lecithin"],
    "Eggs": ["egg", "albumin", "mayonnaise"],
    "Fish / Shellfish": ["fish", "shrimp", "crab", "prawn", "lobster"],
    "Sesame": ["sesame", "tahini", "til"],
    "Sulfites": ["sulfite", "sulphite", "sulfur dioxide"],
    "Mustard": ["mustard", "rai", "sarson"],
    "Corn / Maize": ["corn", "maize", "cornstarch", "corn starch", "maize starch", "corn syrup", "corn flour", "cornflour"],
}

allergen_advice = {
    "Milk / Dairy": "Often present as 'milk solids' or 'whey'. Look for certified 'Dairy-Free' or 'Vegan' alternatives.",
    "Nuts / Peanuts": "May be present via shared machinery. Check for explicit nut-free facility labels.",
    "Gluten / Wheat": "Wheat flour (maida) is the common binder. Opt for certified millet, rice, or chickpea alternatives.",
    "Soy": "Soya lecithin (INS 322) is common in confectionery. Look for sunflower-lecithin or lecithin-free items.",
    "Eggs": "Widely used as structural binders or leavening agents in cakes and cookies.",
    "Fish / Shellfish": "Commonly found hidden in savory pastes and sauces.",
    "Sesame": "Common in Indian snacks and spice mixes.",
    "Sulfites": "Used as preservatives to extend shelf life.",
    "Mustard": "Common in Indian savory namkeens and oils.",
    "Corn / Maize": "Frequently present as corn starch, maize starch, or glucose syrup.",
}


def clean_ocr_text(raw_text):
    text = (raw_text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_ALL_KW = "|".join(re.escape(k) for k in sorted({k for v in allergy_keywords.values() for k in v}, key=len, reverse=True))

# "free from milk", "contains no nuts", "no added soy"  -> these are claims of ABSENCE, not allergens
_FREE_FROM = re.compile(r"\b(?:free\s+(?:from|of)|without|contains\s+no|zero|no)\s+(?:added\s+)?(?:(?:" + _ALL_KW +
                        r")s?\b[\s,/&]*(?:and\s+|or\s+)?)+")
_FREE_WORD = re.compile(r"\b[a-z]+[\s-]*free\b")                      # "gluten free", "dairy-free", "nut free"

# "may contain ...", "traces of ...", "made in a facility that handles ..." -> precaution, not an ingredient
_MAY = re.compile(r"(?:may\s+contain|might\s+contain|traces?\s+of|"
                  r"(?:made|manufactured|produced|processed|packed|packaged)\s+(?:in|on)\s+(?:a\s+|an\s+|the\s+)?"
                  r"(?:facility|factory|premises|line|equipment|shared)|shared\s+(?:facility|equipment|line)|cross[\s-]*contact)"
                  r"(?:(?!\bcontains?\b)[^.]){0,220}")

# Long, distinctive words that OCR often misspells by one letter ("peanuf", "gluien"). Not used for short words.
FUZZY_WORDS = {
    "Nuts / Peanuts": ["peanut", "hazelnut", "pistachio", "almond", "cashew", "walnut", "groundnut"],
    "Gluten / Wheat": ["gluten", "semolina"],
    "Soy": ["soybean", "soyabean"],
    "Eggs": ["albumin"],
    "Fish / Shellfish": ["shellfish", "crustacean"],
    "Sesame": ["sesame", "tahini"],
    "Mustard": ["mustard"],
}
# harmless real words that look like an allergen word (salmon~almond, lutein~gluten, custard~mustard ...)
FUZZY_SKIP = {"salmon", "lutein", "custard", "tahiti", "lactase", "lactone", "sulfate", "sulphate"}


def _has(cleaned, allergy):
    """Exact word match (plurals allowed), then a careful one-letter-off match for distinctive words."""
    for kw in allergy_keywords.get(allergy, []):
        if re.search(r"\b" + re.escape(kw.lower()) + r"s?\b", cleaned):
            return True
    for token in set(cleaned.split()):
        if len(token) < 6 or token in FUZZY_SKIP:
            continue
        for kw in FUZZY_WORDS.get(allergy, []):
            if abs(len(token) - len(kw)) <= 1 and SequenceMatcher(None, token, kw).ratio() >= (0.87 if len(kw) >= 8 else 0.83):
                return True
    return False


# words that look like dairy but are not: "peanut butter" is a nut (not milk), "coconut milk" is not dairy
_NOT_DAIRY_BUTTER = re.compile(r"\b(peanut|cocoa|cacao|shea|almond|cashew|nut|mango|kokum|coconut|illipe)\s+butter\b")
_PLANT_MILK = re.compile(r"\b(coconut|soy|soya|almond|oat|rice|cashew|hazelnut)\s+milk\b")
_CREAM_TARTAR = re.compile(r"\bcream\s+of\s+tartar\b")


def analyze_allergens(raw_text, user_allergies):
    """Returns (contains, may_contain): allergens the pack declares vs. ones it only warns about."""
    t = re.sub(r"\s+", " ", (raw_text or "").lower())
    t = _NOT_DAIRY_BUTTER.sub(r"\1", t)
    t = _PLANT_MILK.sub(r"\1", t)
    t = _CREAM_TARTAR.sub(" ", t)
    t = _FREE_FROM.sub(" ", t)
    t = _FREE_WORD.sub(" ", t)
    may_text = clean_ocr_text(" ".join(m.group(0) for m in _MAY.finditer(t)))
    main_text = clean_ocr_text(_MAY.sub(" . ", t))
    contains = [a for a in user_allergies if _has(main_text, a)]
    may = [a for a in user_allergies if a not in contains and _has(may_text, a)]
    return contains, may


def check_allergens(extracted_text, user_allergies):
    """Kept for older code: every allergen found, whether declared or 'may contain'."""
    contains, may = analyze_allergens(extracted_text, user_allergies)
    return contains + may


# --------------------------------------------------------------------------
# CATEGORY DETECTION (scored, so "chocolate cake" -> cake, not chocolate)
# --------------------------------------------------------------------------
# category: (strong words worth 3 points, weak words worth 1 point)
CATEGORY_RULES = {
    "Cakes & Bakery": (
        "cake|cakes|brownie|muffin|pastry|cupcake|chocopie|choco pie|swiss roll|tiffin|gobbles|bread|bun|rusk toast",
        "sponge|bakery|baked|frosting|cream filled"),
    "Biscuits & Cookies": (
        "biscuit|biscuits|cookie|cookies|cracker|crackers|marie|bourbon|oreo|good day|malkist|digestive|rusk",
        "crunchy|crisp|butter cookies"),
    "Noodles & Pasta": (
        "noodle|noodles|pasta|macaroni|spaghetti|ramen|maggi|maggie|vermicelli|wai wai|yippee",
        "instant|masala tastemaker"),
    "Chips & Snacks": (
        "chips|chip|crisps|namkeen|kurkure|chakli|murukku|bhujia|sev|popcorn|makhana|lays|bingo|banana chips",
        "snack|salted|fried|potato|wafers"),
    "Chocolates & Sweets": (
        "chocolate|munch|perk|kitkat|kit kat|dairy milk|candy|toffee|cocoa bar|5 star|fivestar",
        "sweet|bar|cocoa|wafer"),
    "Ice Cream & Desserts": (
        "ice cream|icecream|kulfi|gelato|sorbet|frozen dessert|falooda",
        "dessert|scoop"),
    "Milk & Beverages": (
        "milk|juice|beverage|drink|shake|lassi|buttermilk|coffee|tea|smoothie",
        "ml|litre|liter|pasteurised|pasteurized"),
    "Sauces & Condiments": (
        "ketchup|sauce|mayonnaise|mayo|spread|jam|dip|chutney|dressing|pickle",
        "tomato|condiment"),
}


# Product/brand words that settle the category on their own (weight 6).
DECISIVE = {
    "Milk & Beverages": "flavoured milk|flavored milk|chocolate milk|milk shake|milkshake|lassi|buttermilk|juice|kool|"
                        "oat milk|almond milk|soy milk|cold coffee|energy drink|soft drink",
    "Noodles & Pasta": "noodles|noodle|wai wai|maggi|maggie|yippee|chowmein|pasta|macaroni|vermicelli|ramen",
    "Chocolates & Sweets": "munch|kitkat|kit kat|dairy milk|five star|5 star|perk|snickers|milky bar|bournville|"
                           "chocolate bar|candy|toffee|eclairs",
    "Ice Cream & Desserts": "ice cream|icecream|kulfi|gelato|frozen dessert",
    "Cakes & Bakery": "cake|cakes|choco pie|chocopie|swiss roll|brownie|muffin|cupcake",
    "Chips & Snacks": "chips|kurkure|namkeen|chakli|murukku|bhujia|lays|bingo|popcorn|banana chips",
    "Biscuits & Cookies": "biscuit|biscuits|cookie|cookies|cracker|malkist|marie|good day|oreo|bourbon|hide and seek",
    "Sauces & Condiments": "ketchup|mayonnaise|chutney|mayo",
}


def _front_text(text):
    """Label text BEFORE the ingredient list - ingredients (e.g. 'milk solids') must not decide the category."""
    cleaned = clean_ocr_text(text)
    idx = cleaned.find("ingredient")
    return cleaned[:idx] if idx >= 15 else cleaned


def deduce_product_category(extracted_text):
    front = _front_text(extracted_text)
    best, best_score = None, 0
    for category, (strong, weak) in CATEGORY_RULES.items():
        score = 6 * len(re.findall(r"\b(?:" + DECISIVE[category] + r")s?\b", front))
        score += 3 * len(re.findall(r"\b(?:" + strong + r")\b", front))
        score += len(re.findall(r"\b(?:" + weak + r")\b", front))
        if score > best_score:
            best, best_score = category, score
    return best or "Biscuits & Cookies"


# --------------------------------------------------------------------------
# PRODUCT RECOGNITION FROM THE PHOTO (Gemini vision) - primary classifier
# --------------------------------------------------------------------------
CATEGORY_HELP = {
    "Cakes & Bakery": "cakes, cupcakes, muffins, brownies, choco pies, swiss rolls, breads, buns, pastries",
    "Biscuits & Cookies": "biscuits, cookies, crackers, rusks (Marie, Good Day, Oreo, Malkist, Parle-G)",
    "Noodles & Pasta": "instant noodles (Maggi, Wai Wai, Yippee, Ching's), pasta, macaroni, vermicelli",
    "Chips & Snacks": "chips, wafers, namkeen, chakli, bhujia, popcorn, makhana, banana chips, Kurkure",
    "Chocolates & Sweets": "chocolate bars, wafer-chocolate bars (Munch, KitKat, Dairy Milk, 5 Star), candy, toffee, mithai",
    "Ice Cream & Desserts": "ice cream, kulfi, frozen desserts",
    "Milk & Beverages": "plain or flavoured milk, milkshakes, lassi, juices, soft drinks, plant milks",
    "Sauces & Condiments": "ketchup, mayonnaise, sauces, jams, spreads, chutneys, dressings",
}
_vision_cache = {}


def _image_bytes(image, max_side=1024):
    img = image.convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def classify_with_vision(image):
    """Asks Gemini to look at the pack photo. Returns {'category','product_name'} or None."""
    global _working_model
    try:
        data = _image_bytes(image)
    except Exception:
        return None
    key = hashlib.sha1(data).hexdigest()
    if key in _vision_cache:
        return _vision_cache[key]
    menu = "\n".join(f"- {c}: {h}" for c, h in CATEGORY_HELP.items())
    prompt = f"""Look at this photo of a packaged food product sold in India and identify what the product IS.
Pick exactly ONE category from this list (copy the name exactly):
{menu}

Rules: judge the product itself, not its ingredients. Flavoured milk is a beverage even if chocolate-flavoured.
A chocolate bar that lists milk is still a chocolate. A noodle pack with a masala/sauce sachet is noodles.
If it fits none of the categories use "Other".

Return ONLY JSON: {{"brand": "...", "product_name": "...", "category": "..."}}"""
    for model in ([_working_model] if _working_model else []) + PREFERRED_MODELS:
        try:
            resp = get_client().models.generate_content(
                model=model,
                contents=[types.Part.from_bytes(data=data, mime_type="image/jpeg"), prompt],
                config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json"))
            d = _extract_json(resp.text)
            cat = str(d.get("category", "")).strip()
            if cat in CATEGORY_HELP:
                name = f"{d.get('brand', '')} {d.get('product_name', '')}".strip()
                _working_model = model
                _vision_cache[key] = {"category": cat, "product_name": name}
                return _vision_cache[key]
            return None                                   # "Other" / invalid -> use text rules
        except Exception:                                 # noqa: BLE001
            continue
    return None


def determine_category(image, extracted_text):
    """Vision first, text rules as backup. Records how the decision was made."""
    rule_cat = deduce_product_category(extracted_text)
    v = classify_with_vision(image) if image is not None else None
    if v:
        LAST_STATUS["detected"] = f"{v['product_name'] or 'product'} -> {v['category']} (photo recognition; text rules said: {rule_cat})"
        return v["category"], v["product_name"]
    LAST_STATUS["detected"] = f"{rule_cat} (text rules - photo recognition unavailable)"
    return rule_cat, ""


def product_hint(extracted_text, max_chars=250):
    """First readable lines of the label (usually brand + product name)."""
    lines = [l.strip() for l in (extracted_text or "").splitlines()]
    lines = [l for l in lines if len(re.findall(r"[A-Za-z]", l)) >= 3]
    return " | ".join(lines)[:max_chars]


# --------------------------------------------------------------------------
# GEMINI RECOMMENDER  (grounded with Google Search - no answers from memory)
# --------------------------------------------------------------------------
# Tried in order. Edit this list if you know a model that works for your key.
PREFERRED_MODELS = [
    "gemini-3.5-flash-lite",   # confirmed working on your key
    "gemini-3.1-flash-lite",   # confirmed working on your key
    "gemini-3.8-flash",
    "gemini-3.5-flash",
]

# Google Search grounding returned "429 quota exceeded" on your key. Set this to True only after
# `python test_quota.py` shows the "search" lines as OK (e.g. after enabling billing).
USE_GOOGLE_SEARCH = False

# False = if Google Search grounding is unavailable, use the checked offline list
#         (safe, no hallucinated products).
# True  = fall back to answers from Gemini's memory (can hallucinate products).
ALLOW_UNGROUNDED = False

_working_model = None
_discovered = None
_search_disabled_until = 0.0


def _discover_models():
    """Asks the API which flash models this key can actually use."""
    global _discovered
    if _discovered is not None:
        return _discovered
    found = []
    try:
        for m in get_client().models.list():
            name = m.name.replace("models/", "")
            actions = getattr(m, "supported_actions", None) or []
            bad = ("image", "tts", "live", "audio", "embedding", "preview", "exp", "thinking", "omni")
            if "flash" in name and "generateContent" in actions and not any(b in name for b in bad):
                found.append(name)
    except Exception:
        pass
    _discovered = sorted(found, reverse=True)
    return _discovered


def _build_prompt(category, scanned_hint, avoid):
    avoid_str = ", ".join(avoid)
    return f"""You are a careful food-product researcher for Indian consumers.

The user scanned this packaged product: "{scanned_hint}"
Product type: {category}
The user is ALLERGIC to: [{avoid_str}]

STEP 1 - Use Google Search to find 4 REAL products of the SAME TYPE ({category}) that are currently listed for
sale in India (amazon.in, bigbasket, blinkit, zepto, flipkart, nykaa, or the brand's own website).
STEP 2 - For each product, read what the listing/brand page says about allergens and ingredients.

STRICT RULES
- ONLY include a product if you actually found its listing in your search results. NEVER invent or guess a
  product. If you can find fewer than 4 real products, return fewer.
- "product_name" must be the real product title as shown on the listing (brand + product).
- Same product type as scanned: a cake gets cakes, muffins, brownies or cake mixes; noodles get noodles; etc.
  Do not suggest cereals, muesli, bars or other product types.
- Gluten-free does NOT mean egg-free or dairy-free. Judge EACH of the user's allergens separately.
- "free_of" may list ONLY allergens (from the user's list above) that the listing/brand page explicitly states
  the product is free of, or clearly does not contain based on its ingredient list. If it is not stated, leave
  that allergen out. Do not assume.
- Prefer healthier options (no maida, less sugar, millet/oats/ragi/jaggery) when they exist.

Return ONLY this JSON (no markdown):
{{"products": [{{"brand": "...", "product_name": "...", "free_of": ["allergen names exactly as in the user's list"], "note": "one short factual sentence from the listing"}}]}}"""


def _extract_json(text):
    raw = (text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"```\s*$", "", raw).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in response")
    return json.loads(raw[start:end + 1])


def _parse_products(raw_text):
    data = _extract_json(raw_text)
    items = data.get("products", []) if isinstance(data, dict) else data
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        brand = str(item.get("brand", "")).strip()
        name = str(item.get("product_name", "")).strip()
        full = name if (not brand or brand.lower() in name.lower()) else f"{brand} {name}"
        full = re.sub(r"\s+", " ", full).strip()
        free_of = item.get("free_of") or []
        if isinstance(free_of, str):
            free_of = [free_of]
        if full:
            out.append({"name": full, "free_of": [str(x).strip() for x in free_of],
                        "note": str(item.get("note", "")).strip()})
    return out


def _grounding_info(resp):
    """Pulls the web-search evidence out of a grounded Gemini response."""
    info = {"queries": [], "supported": "", "sources": []}
    try:
        gm = resp.candidates[0].grounding_metadata
    except Exception:
        return info
    if not gm:
        return info
    info["queries"] = list(gm.web_search_queries or [])
    info["supported"] = " ".join(
        (s.segment.text or "") for s in (gm.grounding_supports or []) if getattr(s, "segment", None))
    for ch in (gm.grounding_chunks or []):
        if getattr(ch, "web", None):
            info["sources"].append({"title": ch.web.title or ch.web.domain or "source", "uri": ch.web.uri or ""})
    return info


_GENERIC = {"free", "gluten", "vegan", "dairy", "egg", "eggless", "pack", "mix", "natural", "organic",
            "healthy", "original", "premium", "the", "and", "with", "sugar", "no", "added"}


def _is_supported(name, supported_text):
    """True only if the BRAND and most distinctive words of the product name appear in text that
    Google Search results back up. Generic words ('gluten free') alone never count."""
    tokens = clean_ocr_text(name).split()
    if not tokens or not supported_text:
        return False
    hay = clean_ocr_text(supported_text)
    if tokens[0] not in hay:                      # first word of 'Brand Product' = the brand
        return False
    distinct = [t for t in tokens if len(t) > 2 and t not in _GENERIC]
    if not distinct:
        return False
    return sum(1 for t in distinct if t in hay) / len(distinct) >= 0.75


def _name_conflicts(name, avoid):
    """True if a product NAME still mentions an allergen (ignores 'gluten free', 'dairy-free', ...)."""
    text = clean_ocr_text(re.sub(r"\b\w+[\s-]*free\b", " ", name, flags=re.I))
    for allergy in avoid:
        for kw in allergy_keywords.get(allergy, []):
            if kw in ("starch", "nut", "til", "rai", "malt"):   # too ambiguous in a product name
                continue
            if re.search(r"\b" + re.escape(kw) + r"s?\b", text):
                return True
    return False


def _type_matches(name, category):
    strong = CATEGORY_RULES.get(category, ("", ""))[0]
    return bool(strong) and bool(re.search(r"\b(?:" + strong + r")s?\b", clean_ocr_text(name)))


def _canon_allergen(label, avoid):
    """Maps a model-written allergen ('dairy', 'gluten') onto the user's own allergy names."""
    lab = clean_ocr_text(label)
    for a in avoid:
        words = [w for w in clean_ocr_text(a.replace("/", " ")).split() if len(w) > 2]
        if lab and (lab in clean_ocr_text(a) or any(w in lab or lab in w for w in words)):
            return a
        if any(re.search(r"\b" + re.escape(k) + r"s?\b", lab) for k in allergy_keywords.get(a, [])):
            return a
    return None


def _finish(products, avoid, category, info):
    """Adds safety text + ranking. Products with confirmed allergen claims come first."""
    ranked = []
    for p in products:
        confirmed = {c for c in (_canon_allergen(x, avoid) for x in p["free_of"]) if c}
        missing = [a for a in avoid if a not in confirmed]
        verified = _is_supported(p["name"], info["supported"])
        bits = []
        if confirmed:
            bits.append("✅ Listing states free of: " + ", ".join(sorted(confirmed)) + ".")
        if missing:
            bits.append("⚠️ Not confirmed for: " + ", ".join(missing) + " - check the label.")
        if p["note"]:
            bits.append(p["note"])
        p.update(reason=" ".join(bits), verified=verified, n_confirmed=len(confirmed),
                 source="Gemini + Google Search" + (" ✓" if verified else " (not individually confirmed)"))
        ranked.append(p)
    ranked.sort(key=lambda p: (not p["verified"], not _type_matches(p["name"], category), -p["n_confirmed"]))
    return ranked


CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rec_cache.json")


def _cache_key(category, avoid):
    return category + "|" + ",".join(sorted(avoid))


def _cache_load():
    try:
        with open(CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _cache_save(key, products):
    try:
        data = _cache_load()
        data[key] = products
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
    except Exception:
        pass


def query_gemini_recommender(category, detected_allergens, user_allergies, extracted_text=""):
    global _working_model, _search_disabled_until
    avoid = list(dict.fromkeys(list(user_allergies) + list(detected_allergens)))
    if not USE_GOOGLE_SEARCH or time.time() < _search_disabled_until:
        return []
    key = _cache_key(category, avoid)
    cached = _cache_load().get(key)
    if cached:                                   # same product type + same allergies seen before
        LAST_STATUS.update(engine="Cache (earlier web-checked result)", model=None, error=None,
                           queries=[], sources=[])
        return [dict(p, source="Cached " + p.get("source", "")) for p in cached]
    prompt = _build_prompt(category, product_hint(extracted_text) or category, avoid)
    search_tool = types.Tool(google_search=types.GoogleSearch())

    chain = ([_working_model] if _working_model else []) + PREFERRED_MODELS
    last_error, tried, discovered_added, i = None, set(), False, 0
    modes = [True] + ([False] if ALLOW_UNGROUNDED else [])

    while i < len(chain):
        model = chain[i]
        i += 1
        if model in tried:
            continue
        tried.add(model)
        for grounded in modes:
            for attempt in range(2):                          # one retry for 429/503
                try:
                    cfg = types.GenerateContentConfig(
                        temperature=0.2,
                        tools=[search_tool] if grounded else None,
                        response_mime_type=None if grounded else "application/json",
                    )
                    resp = get_client().models.generate_content(model=model, contents=prompt, config=cfg)
                    info = _grounding_info(resp) if grounded else {"queries": [], "supported": "", "sources": []}
                    if grounded and not (info["queries"] or info["sources"]):
                        last_error = f"{model}: answered without using Google Search (rejected)"
                        break
                    products = [p for p in _parse_products(resp.text) if not _name_conflicts(p["name"], avoid)]
                    if grounded:
                        products = _finish(products, avoid, category, info)
                        good = [p for p in products if p["verified"]]
                        products = good + ([p for p in products if not p["verified"]] if len(good) < 2 else [])
                    else:
                        for p in products:
                            p.update(reason="⚠️ Gemini answered from memory (not web-checked) - verify it exists.",
                                     source="Gemini (UNVERIFIED)", verified=False)
                    if products:
                        _working_model = model
                        LAST_STATUS.update(engine="Gemini + Google Search" if grounded else "Gemini (ungrounded)",
                                           model=model, error=None, queries=info["queries"],
                                           sources=info["sources"][:8])
                        if grounded and any(p.get("verified") for p in products):
                            _cache_save(key, products[:3])
                        return products[:3]
                    last_error = f"{model}: no usable products"
                    break
                except Exception as e:                        # noqa: BLE001
                    last_error = f"{model}: {type(e).__name__}: {str(e)[:200]}"
                    if grounded and "RESOURCE_EXHAUSTED" in str(e):
                        _search_disabled_until = time.time() + 600     # stop hammering a spent quota
                    quota = "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower()
                    transient = (not quota) and any(c in str(e) for c in ("503", "500", "504", "DEADLINE", "timed out", "Timeout"))
                    if transient and attempt == 0:
                        time.sleep(1.5)
                        continue
                    break

        if i >= len(chain) and not discovered_added:          # ask the API what this key can use
            discovered_added = True
            chain.extend(_discover_models())

    LAST_STATUS.update(engine="Offline list", model=None, error=last_error, queries=[], sources=[])
    return []


# --------------------------------------------------------------------------
# VERIFIED CATALOG  (real products only - Gemini can pick from it but cannot invent)
# --------------------------------------------------------------------------
# free_of  = allergens the brand/listing EXPLICITLY states the product is free of (shown as ✅)
# contains = allergens it contains or may contain -> product is HIDDEN from users with that allergy
# Everything not listed in free_of is shown as "⚠️ not confirmed" so the user checks the label.
# "src" = where I confirmed the product/brand exists (live web search, Oct 2026).
# To add more products, append entries here or put them in my_products.json (same fields).
GLU, MILK, EGG, NUT, SOY, SES = ("Gluten / Wheat", "Milk / Dairy", "Eggs", "Nuts / Peanuts", "Soy", "Sesame")

VERIFIED_CATALOG = [
    # ---- Cakes & Bakery
    {"id": "cake1", "category": "Cakes & Bakery", "name": "Wheafree Gluten Free Cake Mix",
     "free_of": [GLU], "contains": [],
     "note": "No-added-sugar premix; you add the eggs/milk yourself, so substitutes are possible.", "src": "amazon.in listing"},
    {"id": "cake2", "category": "Cakes & Bakery", "name": "Wheafree Gluten Free Cake Rusk",
     "free_of": [GLU], "contains": [EGG],
     "note": "Gluten-free but egg-based.", "src": "retailer listing"},
    {"id": "cake3", "category": "Cakes & Bakery", "name": "Fidalgo's Gluten Free Premium Multigrain Cake Rusk",
     "free_of": [GLU], "contains": [],
     "note": "Gluten-free multigrain cake rusk (200 g); check the label for egg/milk.", "src": "magicdrop.in"},
    # ---- Biscuits & Cookies
    {"id": "bis1", "category": "Biscuits & Cookies", "name": "Wheafree Gluten Free Strawberry Cream Biscuits",
     "free_of": [GLU], "contains": [],
     "note": "Gluten-free cream biscuits; check the label for milk.", "src": "indiamart listing"},
    {"id": "bis2", "category": "Biscuits & Cookies", "name": "Wheafree Gluten Free Jam Biscuits",
     "free_of": [GLU], "contains": [],
     "note": "Gluten-free jam biscuits.", "src": "wheafree.com"},
    # ---- Noodles & Pasta
    {"id": "noo1", "category": "Noodles & Pasta", "name": "Wheafree Gluten Free Rice Noodles",
     "free_of": [GLU], "contains": [],
     "note": "Gluten-free rice noodles.", "src": "wheafree.com"},
    {"id": "noo2", "category": "Noodles & Pasta", "name": "Slurrp Farm Millet Noodles (Foxtail / Little Millet)",
     "free_of": [], "contains": [GLU],
     "note": "No-maida millet noodles with no preservatives; some variants may include wheat flour.", "src": "mishry.com review"},
    {"id": "noo3", "category": "Noodles & Pasta", "name": "Slurrp Farm Vegan Gluten-Free Pasta (Fusilli / Macaroni)",
     "free_of": [MILK, EGG, GLU], "contains": [],
     "note": "Listed as vegan, no-maida, gluten-free multigrain pasta.", "src": "bigbasket.com"},
    {"id": "bis3", "category": "Biscuits & Cookies", "name": "Slurrp Farm Choco Ragi Cookies",
     "free_of": [], "contains": [GLU, MILK],
     "note": "No maida or refined sugar, but contains whole wheat and skimmed milk powder.", "src": "swiggy instamart listing"},
    # ---- Chips & Snacks
    {"id": "snk1", "category": "Chips & Snacks", "name": "Wheafree Gluten Free Bikaneri Bhujia",
     "free_of": [GLU], "contains": [],
     "note": "Gluten-free namkeen.", "src": "wheafree.com"},
    {"id": "snk2", "category": "Chips & Snacks", "name": "Wheafree Gluten Free Cheese Makhana",
     "free_of": [GLU], "contains": [MILK],
     "note": "Gluten-free but cheese-flavoured (dairy).", "src": "wheafree.com"},
    {"id": "snk3", "category": "Chips & Snacks", "name": "Haldiram's Soya Sticks (Chatpata Masala)",
     "free_of": [], "contains": [SOY],
     "note": "Ingredients printed on the pack: tapioca starch, palmolein oil, soya flour, urad flour, spices. "
             "Check the pack's 'may contain' line.", "src": "ingredient list read from the pack photo"},
    # ---- Chocolates & Sweets
    {"id": "cho1", "category": "Chocolates & Sweets", "name": "The Whole Truth 71% Dark Chocolate (date-sweetened)",
     "free_of": [], "contains": [],
     "note": "Dark chocolate sweetened with dates; check 'may contain' warnings.", "src": "nykaa.com / swiggy instamart"},
    {"id": "cho2", "category": "Chocolates & Sweets", "name": "Darkins Vegan & Gluten-Free Artisanal Dark Chocolate",
     "free_of": [MILK, EGG, GLU], "contains": [],
     "note": "Listed as vegan and gluten-free; some flavours contain almonds, so check the flavour.", "src": "bigbasket.com"},
    {"id": "cho3", "category": "Chocolates & Sweets", "name": "Piperleaf Vegan Mylk Chocolate",
     "free_of": [MILK, EGG], "contains": [NUT],
     "note": "Vegan 'mylk' chocolate made with hazelnut and cashew paste.", "src": "vegnews.com"},
    {"id": "cho4", "category": "Chocolates & Sweets", "name": "MilkinOats Vegan Oat Milk Chocolate",
     "free_of": [MILK, EGG], "contains": [NUT],
     "note": "Plant-based milk chocolate made with oat milk, cashews and almonds.", "src": "greenqueen.com.hk"},
    # ---- Ice Cream & Desserts
    {"id": "ice1", "category": "Ice Cream & Desserts", "name": "WhiteCub Vegan Ice Cream",
     "free_of": [MILK, EGG], "contains": [],
     "note": "India's first dedicated vegan ice cream brand; free from dairy, eggs and honey.", "src": "indianretailer.in"},
    {"id": "ice2", "category": "Ice Cream & Desserts", "name": "Minus.30 Vegan Ice Cream",
     "free_of": [MILK, SOY], "contains": [],
     "note": "Vegan range described as soy-free; check flavour for nuts.", "src": "indianretailer.in"},
    {"id": "ice3", "category": "Ice Cream & Desserts", "name": "Nomou Vegan Gelato",
     "free_of": [MILK], "contains": [NUT, SES],
     "note": "Plant-based (coconut milk / palm sugar); some flavours use nuts or sesame.", "src": "thebetterindia.com"},
    {"id": "ice4", "category": "Ice Cream & Desserts", "name": "The Brooklyn Creamery Vegan Ice Cream",
     "free_of": [MILK], "contains": [NUT],
     "note": "Vegan range; plant bases are often nut-based, so check the flavour.", "src": "yourstory.com"},
    # ---- Milk & Beverages
    {"id": "mlk1", "category": "Milk & Beverages", "name": "Country Delight Oat Beverage",
     "free_of": [MILK, SOY, NUT], "contains": [],
     "note": "Oat drink made in an allergen-controlled facility; oats - check gluten-free status if coeliac.", "src": "vegconomist.com"},
    {"id": "mlk2", "category": "Milk & Beverages", "name": "Epigamia Unsweetened Almond Milk",
     "free_of": [MILK, GLU], "contains": [NUT],
     "note": "Vegan, dairy-free, gluten-free almond milk.", "src": "brand listing"},
    {"id": "mlk3", "category": "Milk & Beverages", "name": "MilkinOats Oat Milk",
     "free_of": [MILK], "contains": [],
     "note": "Plant-based oat milk, dairy-free.", "src": "indianretailer.in"},
]


def _load_catalog():
    items = list(VERIFIED_CATALOG)
    extra = os.path.join(os.path.dirname(os.path.abspath(__file__)), "my_products.json")
    try:
        with open(extra, encoding="utf-8") as f:
            for i, p in enumerate(json.load(f)):
                p.setdefault("id", f"user{i}")
                p.setdefault("free_of", [])
                p.setdefault("contains", [])
                p.setdefault("note", "")
                p.setdefault("src", "added by you")
                items.append(p)
    except Exception:
        pass
    return items


def _catalog_candidates(category, avoid):
    """Catalog products of this category that do not contain any of the user's allergens."""
    out = []
    for p in _load_catalog():
        if p["category"] != category or any(a in avoid for a in p["contains"]):
            continue
        confirmed = [a for a in avoid if a in p["free_of"]]
        missing = [a for a in avoid if a not in p["free_of"]]
        out.append(dict(p, confirmed=confirmed, missing=missing))
    out.sort(key=lambda p: -len(p["confirmed"]))
    return out


def _describe(p):
    bits = []
    if p["confirmed"]:
        bits.append("✅ Brand states free of: " + ", ".join(p["confirmed"]) + ".")
    if p["missing"]:
        bits.append("⚠️ Not confirmed for: " + ", ".join(p["missing"]) + " - check the label.")
    bits.append(p["note"])
    return " ".join(bits)


def _gemini_pick(category, scanned_hint, avoid, candidates):
    global _working_model
    """Gemini chooses the best matches from the verified list BY ID (so it cannot invent a product)."""
    listing = "\n".join(
        f'- id={c["id"]} | {c["name"]} | confirmed free of: {", ".join(c["confirmed"]) or "none"} | {c["note"]}'
        for c in candidates)
    prompt = f"""A shopper allergic to [{", ".join(avoid)}] scanned this product: "{scanned_hint}" ({category}).
Choose up to 3 of the CANDIDATES below that are the best safe alternatives. Prefer products with more
confirmed allergen-free claims, then healthier options. Use ONLY ids from the list - never invent products.

CANDIDATES:
{listing}

Return ONLY JSON: {{"picks": [{{"id": "...", "reason": "one short sentence"}}]}}"""
    last_error = None
    for model in ([_working_model] if _working_model else []) + PREFERRED_MODELS:
        try:
            resp = get_client().models.generate_content(
                model=model, contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2, response_mime_type="application/json"))
            picks = _extract_json(resp.text).get("picks", [])
            by_id = {c["id"]: c for c in candidates}
            chosen = []
            for pk in picks:
                c = by_id.get(str(pk.get("id", "")).strip())
                if c and c not in [x[0] for x in chosen]:
                    chosen.append((c, str(pk.get("reason", "")).strip()))
            if chosen:
                _working_model = model
                LAST_STATUS.update(engine="Gemini selected from verified catalog", model=model, error=None,
                                   queries=[], sources=[])
                return chosen[:3]
            last_error = f"{model}: no valid picks"
        except Exception as e:                                   # noqa: BLE001
            last_error = f"{model}: {type(e).__name__}: {str(e)[:160]}"
    LAST_STATUS.update(engine="Verified catalog (rule-based)", model=None, error=last_error, queries=[], sources=[])
    return []


def catalog_recommend(category, avoid, extracted_text, limit=3):
    candidates = _catalog_candidates(category, avoid)
    if not candidates:
        return []
    picks = _gemini_pick(category, product_hint(extracted_text) or category, avoid, candidates[:8])
    if not picks:                                                # Gemini unavailable -> best-scored candidates
        picks = [(c, "") for c in candidates[:limit]]
    elif len(picks) < 2:                                         # Gemini chose only one -> add the next best
        chosen = {c["id"] for c, _ in picks}
        picks += [(c, "") for c in candidates if c["id"] not in chosen][:2 - len(picks)]
    engine = LAST_STATUS["engine"] or ""
    out = []
    for c, reason in picks:
        text = _describe(c) + (f" {reason}" if reason else "")
        out.append({"name": c["name"], "reason": text, "verified": True,
                    "source": ("AI-selected · " if engine.startswith("Gemini") else "") + f"Verified catalog ({c['src']})"})
    return out


SEARCH_NOUN = {
    "Cakes & Bakery": "cake", "Biscuits & Cookies": "cookies", "Noodles & Pasta": "noodles",
    "Chips & Snacks": "snacks", "Chocolates & Sweets": "dark chocolate", "Ice Cream & Desserts": "ice cream",
    "Milk & Beverages": "plant based milk", "Sauces & Condiments": "ketchup",
}


def search_suggestions(category, avoid, n):
    """Live Amazon.in searches for the right product TYPE. These are links, not product claims, so they can't be fake."""
    noun = SEARCH_NOUN.get(category, "food")
    free = " ".join(a.split("/")[0].strip().lower() + " free" for a in avoid)
    first = (avoid[0].split("/")[0].strip().lower() + " free") if avoid else "allergen free"
    queries = [f"{free} {noun}".strip(),
               f"vegan {noun}" if (MILK in avoid or EGG in avoid) else f"{noun} allergen free",
               f"{first} {noun}"]
    out, seen = [], set()
    for q in queries:
        if q in seen:
            continue
        seen.add(q)
        out.append({"name": f"Search online: {q}", "query": q, "verified": False,
                    "reason": f"No verified {noun} match your allergies in our list, so this opens a live search "
                              "for the right product type. Read the label before buying.",
                    "source": "Search suggestion"})
        if len(out) >= n:
            break
    return out


# --------------------------------------------------------------------------
# MAIN ENTRY POINT (same signature app.py already uses)
# --------------------------------------------------------------------------
def get_safe_alternatives(extracted_text, user_allergies, detected_allergens=None, image=None):
    if detected_allergens is None:
        detected_allergens = check_allergens(extracted_text, user_allergies)

    insights = [
        f"**{a}**: {allergen_advice.get(a, 'Check the ingredient label carefully for cross-contamination.')}"
        for a in detected_allergens
    ]

    category, _scanned_name = determine_category(image, extracted_text)
    avoid = list(dict.fromkeys(list(user_allergies) + list(detected_allergens)))
    free_terms = " ".join(a.split("/")[0].strip().lower() + " free" for a in avoid)
    search_term = f"{free_terms} {category}".strip()

    # 1) optional live-web path (only if USE_GOOGLE_SEARCH is on and the quota allows)
    safe_products = query_gemini_recommender(category, detected_allergens, user_allergies, extracted_text)

    # 2) verified catalog, with Gemini choosing among real products
    if len(safe_products) < 2:
        have = {p["name"] for p in safe_products}
        for p in catalog_recommend(category, avoid, extracted_text):
            if p["name"] not in have:
                safe_products.append(p)

    if len(safe_products) < 2:                   # always show at least 2 suggestions
        safe_products += search_suggestions(category, avoid, 2 - len(safe_products))

    return category, safe_products[:3], insights, search_term
