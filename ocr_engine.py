"""
ocr_engine.py - reads a label photo reliably and quickly. This is the PRIMARY part of SafeBite.

Why it exists: phone photos are ~12 megapixels and have glare, colour backgrounds and curved packs.
The old approach enlarged every image 3x (a 12 MP photo became 108 MP: slow, and it read LESS text).

What it does:
  1. Fixes phone-photo rotation (EXIF) and resizes to a sensible size (never huge, small images are enlarged).
  2. Runs several different "views" of the image through Tesseract and combines the text:
       gray      - plain grayscale with gentle contrast        (always)
       colour    - brightest colour channel; fixes text on yellow/red/coloured packs   (always)
       adaptive  - local-contrast threshold; fixes glare and uneven light              (only if the read is weak)
       inverted  - light text on dark background                                       (only if the image is dark / still weak)
  3. Stops adding passes when the read is good or the time budget is used up.
"""
import time
import numpy as np
import pytesseract
from pytesseract import Output
from PIL import Image, ImageOps, ImageFilter

MAX_SIDE = 2200        # longest side never exceeds this (keeps Tesseract fast)
MIN_SIDE = 1400        # small web images are enlarged up to about this
TIME_BUDGET = 25       # seconds; no new pass starts after this
GOOD_CONF, GOOD_WORDS = 72, 80


def prepare(image):
    """Upright RGB image of a sensible size."""
    img = ImageOps.exif_transpose(image) or image
    img = img.convert("RGB")
    longest = max(img.size)
    scale = MAX_SIDE / longest if longest > MAX_SIDE else (min(MIN_SIDE / longest, 3.0) if longest < MIN_SIDE else 1.0)
    if abs(scale - 1.0) > 0.01:
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.Resampling.LANCZOS)
    return img


def _gray(img):
    return ImageOps.autocontrast(img.convert("L"), cutoff=1).filter(ImageFilter.SHARPEN)


def _colour(img):
    """Brightest channel: dark text stays dark on any bright coloured background."""
    return ImageOps.autocontrast(Image.fromarray(np.asarray(img).max(axis=2).astype("uint8")), cutoff=1)


def _adaptive(img):
    """Compare each pixel with its neighbourhood, so glare and shadows do not wash out text."""
    g = _colour(img)
    arr = np.asarray(g).astype("int16")
    blur = np.asarray(g.filter(ImageFilter.BoxBlur(max(15, g.width // 60)))).astype("int16")
    return Image.fromarray(np.where(arr > blur - 8, 255, 0).astype("uint8"))


def _inverted(img):
    return ImageOps.invert(_colour(img))


def _ocr(img, psm):
    """Returns (text, mean word confidence, word count)."""
    d = pytesseract.image_to_data(img, config=f"--oem 3 --psm {psm}", output_type=Output.DICT, timeout=20)
    lines, confs = {}, []
    for i, word in enumerate(d["text"]):
        word = str(word).strip()
        try:
            conf = float(d["conf"][i])
        except (TypeError, ValueError):
            continue
        if not word or conf < 0:
            continue
        lines.setdefault((d["block_num"][i], d["par_num"][i], d["line_num"][i]), []).append(word)
        confs.append(conf)
    text = "\n".join(" ".join(w) for _, w in sorted(lines.items()))
    return text, (sum(confs) / len(confs) if confs else 0.0), len(confs)


def read_label(image, fallback=None):
    """Main entry point. fallback(img)->text is a backup reader used if Tesseract reads almost nothing. Returns {'text', 'confidence', 'words', 'passes', 'seconds', 'error'}."""
    start = time.time()
    out = {"text": "", "confidence": 0.0, "words": 0, "passes": [], "seconds": 0.0, "error": None}
    try:
        img = prepare(image)
        dark = float(np.asarray(img.convert("L")).mean()) < 110
        plan = [("gray", _gray, 6), ("colour", _colour, 6)]
        extras = [("adaptive", _adaptive, 6), ("inverted", _inverted, 6)]
        texts, best_conf, total_words, seen = [], 0.0, 0, set()

        def run(name, fn, psm):
            nonlocal best_conf, total_words
            try:
                text, conf, words = _ocr(fn(img), psm)
            except Exception as e:                              # one failed pass must not stop the others
                out["passes"].append((name, 0.0, 0)); out["error"] = f"{name}: {e}"
                return
            out["passes"].append((name, round(conf, 1), words))
            best_conf, total_words = max(best_conf, conf), max(total_words, words)
            for line in text.split("\n"):                       # combine passes, skipping exact repeats
                key = "".join(ch for ch in line.lower() if ch.isalnum())
                if key and key not in seen:
                    seen.add(key); texts.append(line)

        for name, fn, psm in plan:
            run(name, fn, psm)
        weak = best_conf < GOOD_CONF or total_words < GOOD_WORDS
        for name, fn, psm in extras:
            if time.time() - start > TIME_BUDGET:
                break
            if (name == "adaptive" and weak) or (name == "inverted" and (dark or (weak and best_conf < 60))):
                run(name, fn, psm)
        out.update(text="\n".join(texts), confidence=round(best_conf, 1), words=total_words)

        if len("".join(texts).strip()) < 25 and fallback is not None:          # Tesseract failed or found no text
            try:
                backup = (fallback(img) or "").strip()
            except Exception as e:
                backup = ""
                out["error"] = f"{out['error'] or ''} | backup reader: {e}".strip(" |")
            if len(backup) > 25:
                out.update(text=backup, confidence=80.0, words=len(backup.split()))
                out["passes"].append(("gemini-backup", 80.0, len(backup.split())))
    except Exception as e:
        out["error"] = str(e)
    out["seconds"] = round(time.time() - start, 1)
    return out
