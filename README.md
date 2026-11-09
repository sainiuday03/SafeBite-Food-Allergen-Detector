# SafeBite - Food Allergen Detection and Safe-Alternative Recommender

Upload a photo of a food label. SafeBite reads it with OCR, checks it against your saved allergies, and suggests safer alternatives.

**How it works:** multi-pass Tesseract OCR -> rule-based allergen analysis ("contains" vs "may contain") -> per-allergen XGBoost second opinion (TF-IDF features, 71,382 Open Food Facts products, average F1 0.83) -> Gemini product recognition and selection from a verified product catalog.

**Tech:** Python, Streamlit, Tesseract, scikit-learn, XGBoost, Google Gemini API, SQLite.

**Run locally**
1. Install Tesseract OCR and `pip install -r requirements.txt`
2. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add your Gemini key
3. `streamlit run app.py`

*Prototype for educational use. Not medical advice - always read the pack.*
