"""ml_allergen.py - second opinion from the trained allergen models (never replaces the keyword check)."""
import os
import re
import joblib

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "allergen_models.pkl")
_bundle = None


def _load():
    global _bundle
    if _bundle is None:
        try:
            _bundle = joblib.load(_PATH)
        except Exception:                 # missing file or version mismatch -> app simply skips the ML check
            _bundle = False
    return _bundle


def ml_suspects(label_text, user_allergies, already_found=()):
    """Returns [(allergen, probability)] the models suspect that the keyword check did not already find."""
    bundle = _load()
    if not bundle or not label_text.strip():
        return []
    try:
        match = re.search(r"ingredients?\s*[:\-]?", label_text, re.I)       # models were trained on ingredient lists
        X = bundle["vectorizer"].transform([(label_text[match.end():] if match else label_text).lower()])
        out = []
        for allergy in user_allergies:
            model = bundle["models"].get(allergy)
            if model is None or allergy in already_found:
                continue
            p = float(model.predict_proba(X)[0, 1])
            if p >= bundle["thresholds"][allergy]:
                out.append((allergy, p))
        return sorted(out, key=lambda x: -x[1])
    except Exception:
        return []
