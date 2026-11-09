"""Run once on your laptop:  python make_requirements.py   -> writes requirements.txt with YOUR installed versions."""
import sys
from importlib.metadata import version, PackageNotFoundError

PACKAGES = ["streamlit", "pytesseract", "pillow", "numpy", "scikit-learn", "xgboost", "joblib", "google-genai"]
lines, missing = [], []
for p in PACKAGES:
    try:
        lines.append(f"{p}=={version(p)}")
    except PackageNotFoundError:
        missing.append(p)
open("requirements.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("\nPython version here:", sys.version.split()[0], "-> choose the same minor version in Streamlit Cloud > Advanced settings")
if missing:
    print("NOT INSTALLED (install or remove from the list):", missing)
