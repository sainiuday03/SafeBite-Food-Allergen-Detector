import io
import os
import streamlit as st
import urllib.parse
from PIL import Image, ImageOps
import pytesseract

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="SafeBite", page_icon="🍃", layout="wide")

from database import init_db, register_user, authenticate_user, update_user_allergies
from recommendations import analyze_allergens, get_safe_alternatives, LAST_STATUS
from ocr_engine import read_label
from ml_allergen import ml_suspects

if os.name == "nt":   # only on Windows; a Linux server finds Tesseract by itself
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

init_db()


@st.cache_data(show_spinner=False, max_entries=20)
def cached_ocr(image_bytes):
    """OCR runs once per photo; clicking other buttons no longer repeats it."""
    return read_label(Image.open(io.BytesIO(image_bytes)))


@st.cache_data(show_spinner=False, max_entries=50)
def cached_recommendations(image_bytes, text, allergies, detected):
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes)))
    result = get_safe_alternatives(text, list(allergies), list(detected), image=image)
    return result + (dict(LAST_STATUS),)


def store_buttons(query):
    """Search the product on Amazon and on Google (any store), since no single store sells everything."""
    q = urllib.parse.quote(query)
    c1, c2 = st.columns(2)
    c1.link_button("🛒 Amazon", f"https://www.amazon.in/s?k={q}")
    c2.link_button("🔎 Google", f"https://www.google.com/search?q={q}+buy+online+India")


def show_alternatives(image_bytes, text, allergies, flagged):
    """Safe alternatives for the scanned product - used for 'contains' AND 'may contain' results."""
    with st.spinner("Finding safe alternatives..."):
        category, safe_products, insights, search_term, status = cached_recommendations(
            image_bytes, text, tuple(allergies), tuple(flagged))

    st.markdown("#### 🛡️ Allergen Insights:")
    for info in insights:
        st.info(info)

    st.markdown(f"#### 🛍️ Safe Alternatives ({category}):")
    for prod in safe_products:
        st.success(f"**{prod['name']}**")
        if prod.get("reason"):
            st.caption(f"{prod['reason']}  ·  _{prod.get('source', '')}_")
        store_buttons(prod.get("query", prod["name"]))

    st.divider()
    with st.expander("⚙️ Recommendation engine status"):
        st.write(f"Detected: **{status.get('detected', '')}**")
        st.write(f"Engine: **{status['engine']}**  |  Model: **{status['model']}**")
        if status.get("error"):
            st.code(status["error"])
    st.markdown("**🔍 Find more allergen-free options:**")
    store_buttons(search_term)


ALL_AVAILABLE_ALLERGIES = [
    "Milk / Dairy", "Nuts / Peanuts", "Gluten / Wheat", 
    "Soy", "Eggs", "Fish / Shellfish", "Sesame", "Sulfites",
    "Mustard", "Corn / Maize"
]

# --- 2. CLEAN STYLING ---
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem !important;
        font-weight: 800;
        color: #4CAF50;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1rem;
        color: #888888;
        margin-top: -5px;
        margin-bottom: 15px;
        font-style: italic;
    }
    </style>
""", unsafe_allow_html=True)

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = ""
if 'user_allergies' not in st.session_state:
    st.session_state.user_allergies = []

# --- AUTHENTICATION SCREEN ---
if not st.session_state.logged_in:
    st.markdown('<p class="main-title">SafeBite🍃</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Your Smart Food Allergen Guardian</p>', unsafe_allow_html=True)
    
    auth_option = st.sidebar.radio("Navigation", ["Login", "Register"])

    if auth_option == "Register":
        st.subheader("📝 Create Your Safe Profile")
        with st.form("register_form"):
            reg_username = st.text_input("Choose Username:")
            reg_password = st.text_input("Choose Password:", type="password")
            st.write("**Select your allergens:**")
            
            col1, col2 = st.columns(2)
            chosen_allergies = []
            for i, item in enumerate(ALL_AVAILABLE_ALLERGIES):
                if i % 2 == 0:
                    with col1:
                        if st.checkbox(item): chosen_allergies.append(item)
                else:
                    with col2:
                        if st.checkbox(item): chosen_allergies.append(item)

            if st.form_submit_button("Register Account"):
                if register_user(reg_username.strip(), reg_password.strip(), chosen_allergies):
                    st.success("Account created! Switch to the Login tab.")
                else:
                    st.error("Username already taken or fields missing.")

    elif auth_option == "Login":
        st.subheader("🔑 Welcome Back")
        with st.form("login_form"):
            login_user = st.text_input("Username:")
            login_pass = st.text_input("Password:", type="password")
            
            if st.form_submit_button("Sign In"):
                allergies = authenticate_user(login_user.strip(), login_pass.strip())
                if allergies is not None:
                    st.session_state.logged_in = True
                    st.session_state.current_user = login_user.strip()
                    st.session_state.user_allergies = allergies
                    st.rerun()
                else:
                    st.error("Invalid credentials.")

# --- MAIN DASHBOARD ---
else:
    st.sidebar.header(f"👤 Profile: {st.session_state.current_user}")
    st.sidebar.write("**Your Active Alerts:**")
    for a in st.session_state.user_allergies:
        st.sidebar.markdown(f"- ⚠️ {a}")
        
    st.sidebar.divider()
    
    with st.sidebar.expander("⚙️ Edit Allergies"):
        with st.form("edit_form"):
            updated_allergies = []
            for item in ALL_AVAILABLE_ALLERGIES:
                is_checked = item in st.session_state.user_allergies
                if st.checkbox(item, value=is_checked, key=f"edit_{item}"):
                    updated_allergies.append(item)
                    
            if st.form_submit_button("Save Changes"):
                update_user_allergies(st.session_state.current_user, updated_allergies)
                st.session_state.user_allergies = updated_allergies
                st.success("Profile Updated!")
                st.rerun()
                
    st.sidebar.divider()
    if st.sidebar.button("Log Out"):
        st.session_state.logged_in = False
        st.session_state.current_user = ""
        st.session_state.user_allergies = []
        st.rerun()

    st.markdown('<p class="main-title">SafeBite🍃</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Your Smart Food Allergen Guardian</p>', unsafe_allow_html=True)
    st.write(f"👋 **Welcome, {st.session_state.current_user}!** Upload an ingredient label to scan.")
    
    st.divider()

    # Make sure this line exists right before using left_col / right_col
    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.subheader("📸 Upload Label")
        st.caption("Tip: photograph the INGREDIENTS panel flat, close-up and well lit, with the text filling the frame.")
        uploaded_file = st.file_uploader("Choose product image", type=["jpg", "jpeg", "png"])
        image_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
        if image_bytes:
            image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes)))
            st.image(image, caption="Uploaded Product Label", width="stretch")

    with right_col:
        st.subheader("📊 Scan Results & Recommendations")

        if image_bytes:
            try:
                with st.spinner("Reading the label..."):
                    ocr = cached_ocr(image_bytes)
                extracted_text = ocr["text"]
                allergies = st.session_state.user_allergies

                with st.expander("🔍 View Raw Extracted Text"):
                    st.caption(f"OCR passes: {', '.join(p[0] for p in ocr['passes'])}  |  confidence {ocr['confidence']:.0f}%  |  {ocr['seconds']} s")
                    st.write(extracted_text if extracted_text.strip() else "NO TEXT DETECTED")

                if not extracted_text.strip():
                    st.warning("No text could be read from this photo. Please retake it closer, flatter and in better light.", icon="📷")
                    st.stop()
                if ocr["confidence"] < 55 or ocr["words"] < 25:
                    st.warning("The label was hard to read, so this result may be incomplete. A clearer photo of the "
                               "ingredients panel will give a more reliable answer.", icon="📷")

                contains, may = analyze_allergens(extracted_text, allergies)
                ml_flags = ml_suspects(extracted_text, allergies, contains + may)
                ml_text = ", ".join(f"**{a}** ({p:.0%})" for a, p in ml_flags)
                may_text = "\n".join(f"- **{a}**" for a in may)

                if contains:
                    bullets = "\n".join(f"- **{a}**" for a in contains)
                    st.error(f"**UNSAFE PRODUCT!**\n\nContains your allergens:\n{bullets}", icon="🚫")
                    if may:
                        st.warning(f"**May also contain traces of:**\n{may_text}", icon="⚠️")
                    if ml_flags:
                        st.warning(f"🤖 The AI model also suspects: {ml_text}. Check the label.", icon="⚠️")
                    show_alternatives(image_bytes, extracted_text, allergies, contains)

                elif may or ml_flags:
                    parts = []
                    if may:
                        parts.append(f"The pack says it **may contain traces** of:\n{may_text}")
                    if ml_flags:
                        parts.append(f"The AI model also suspects: {ml_text}.")
                    st.warning("**POSSIBLE ALLERGEN - CHECK THE PACK**\n\n" + "\n\n".join(parts) +
                               "\n\nNot listed as an ingredient, but may be unsafe if you are highly sensitive. "
                               "Here are alternatives to consider:", icon="⚠️")
                    show_alternatives(image_bytes, extracted_text, allergies, may + [a for a, _ in ml_flags])

                else:
                    st.success("**NO REGISTERED ALLERGENS FOUND**\n\nNone of your registered allergens were detected in the scanned text. "
                               "Still check any 'may contain' warnings on the pack.", icon="✅")

            except Exception as e:
                st.error(f"Error processing image: {e}")

        else:
            st.info("👈 Upload an image on the left to see the analysis, safety warnings, and product alternatives appear here instantly!")


# ---------------------------------------------- 2nd ----------------------------------------------

