import streamlit as st
import os
import uuid
from pipeline.runpipeline import run_pipeline
from PIL import Image
import json

# =====================================================
# MUST BE FIRST STREAMLIT COMMAND
# =====================================================
st.set_page_config(page_title="Prescription AI", layout="wide")

# =====================================================
# SESSION STATE INITIALIZATION (CRITICAL)
# =====================================================
if "processed" not in st.session_state:
    st.session_state.processed = False

if "result" not in st.session_state:
    st.session_state.result = None

if "blister_paths" not in st.session_state:
    st.session_state.blister_paths = []

# =====================================================
# 🌐 LANGUAGE SELECTION (NEW)
# =====================================================
st.header("🌐 Select Language")

language = st.radio(
    "Choose your preferred language:",
    ["English", "Tamil"]
)

# =====================================================
# STYLES
# =====================================================
st.markdown("""
<style>
body {
    background-color: #f4f8fb;
}
.big-title {
    font-size: 34px;
    font-weight: 700;
    color: #1f4fd8;
}
.subtitle {
    font-size: 16px;
    color: #4a5568;
    margin-bottom: 20px;
}
.card {
    padding: 18px;
    border-radius: 14px;
    margin-bottom: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}
.schedule-card {
    background: linear-gradient(135deg, #e8f0fe, #f7fbff);
    border-left: 6px solid #4c6ef5;
}
.success {
    background-color: #e6fffa;
    border-left: 6px solid #38b2ac;
}
.warning {
    background-color: #fff7e6;
    border-left: 6px solid #f6ad55;
}
.error {
    background-color: #ffecec;
    border-left: 6px solid #e53e3e;
}
.time-pill {
    display: inline-block;
    padding: 6px 12px;
    margin: 4px 6px 4px 0;
    background-color: #edf2ff;
    border-radius: 20px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# DIRECTORIES
# =====================================================
UPLOAD_DIR = "uploads"
PRES_DIR = os.path.join(UPLOAD_DIR, "prescriptions")
BLISTER_DIR = os.path.join(UPLOAD_DIR, "blisters")

os.makedirs(PRES_DIR, exist_ok=True)
os.makedirs(BLISTER_DIR, exist_ok=True)

# =====================================================
# HEADER
# =====================================================
st.markdown('<div class="big-title">💊 Prescription Interpretation & Scheduler</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Helps patients understand when to take medicines and verifies tablets safely.</div>', unsafe_allow_html=True)

# =====================================================
# PRESCRIPTION UPLOAD
# =====================================================
st.header("1️⃣ Upload Prescription Image")

pres_file = st.file_uploader(
    "Prescription Image",
    type=["jpg", "jpeg", "png"],
    key="pres_uploader"
)

pres_path = None
if pres_file:
    pres_filename = f"{uuid.uuid4()}_{pres_file.name}"
    pres_path = os.path.join(PRES_DIR, pres_filename)

    with open(pres_path, "wb") as f:
        f.write(pres_file.read())

    st.image(Image.open(pres_path), caption="📄 Uploaded Prescription", width=320)

# =====================================================
# BLISTER UPLOAD
# =====================================================
st.header("2️⃣ Upload Medicine Strip Images (Tablets only)")

blister_files = st.file_uploader(
    "Blister Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="blister_uploader"
)

if blister_files:
    st.session_state.blister_paths = []

    for bf in blister_files:
        filename = f"{uuid.uuid4()}_{bf.name}"
        path = os.path.join(BLISTER_DIR, filename)

        with open(path, "wb") as f:
            f.write(bf.read())

        st.session_state.blister_paths.append(path)

    st.success(f"{len(st.session_state.blister_paths)} blister image(s) uploaded")

# =====================================================
# RUN PIPELINE
# =====================================================
st.header("3️⃣ Run Analysis")

if st.button("🚀 Process Prescription"):
    if not pres_path:
        st.error("Please upload a prescription image")
    else:
        with st.spinner("Processing..."):
            st.session_state.result = run_pipeline(
                pres_path,
                st.session_state.blister_paths
            )

            st.session_state.processed = True

            # ✅ SAVE WITH LANGUAGE
            os.makedirs("output", exist_ok=True)

            data_to_save = {
                "schedule": st.session_state.result["schedule"],
                "language": language
            }

            with open("output/schedule.json", "w") as f:
                json.dump(data_to_save, f, indent=4)

        st.success("✅ Processing completed")
        st.info("Schedule + language saved")

# =====================================================
# DISPLAY OUTPUT
# =====================================================
if st.session_state.processed and st.session_state.result:

    result = st.session_state.result

    st.subheader("📅 Your Medicine Schedule")

    schedule = result.get("schedule", [])

    if schedule:
        for med in schedule:
            time_html = "".join(
                [f"<span class='time-pill'>⏰ {t}</span>" for t in med["times"]]
            )

            st.markdown(f"""
            <div class="card schedule-card">
                <h4>💊 {med['medicine']}</h4>
                <b>Duration:</b> {med['duration_days']} day(s)<br><br>
                {time_html}
            </div>
            """, unsafe_allow_html=True)

 # ---------------- VALIDATION ----------------
    st.subheader("🔍 Tablet Verification")

    validation = result.get("validation", {})

    if validation.get("matched"):
        st.markdown("### ✅ Verified Tablets")
        for m in validation["matched"]:
            st.markdown(f"""
            <div class="card success">
                ✔ <b>{m['generic_name']}</b><br>
                Brand: {m['brand_name']}
            </div>
            """, unsafe_allow_html=True)

    if validation.get("unmatched_prescription"):
        st.markdown("### ⚠ Tablets in Prescription but No Strip Uploaded")
        for m in validation["unmatched_prescription"]:
            st.markdown(f"""
            <div class="card warning">
                ⚠ {m['medicine_name']}
            </div>
            """, unsafe_allow_html=True)

    if validation.get("unmatched_blisters"):
        st.markdown("### ⚠ Extra Uploaded Tablet Strips")
        for b in validation["unmatched_blisters"]:
            st.markdown(f"""
            <div class="card warning">
                ⚠ {b['generic_name']} ({b['brand_name']})
            </div>
            """, unsafe_allow_html=True)

    if validation.get("needs_manual_review"):
        st.error("🔴 Manual review recommended due to mismatch.")
    else:
        st.success("🟢 Tablet verification completed successfully.")


# =====================================================
# RESET BUTTON
# =====================================================
if st.button("🔄 Reset App"):
    for key in [
        "processed",
        "result",
        "blister_paths",
        "pres_uploader",
        "blister_uploader"
    ]:
        if key in st.session_state:
            del st.session_state[key]

    st.experimental_rerun()

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<hr>
<center>
<small>⚕️ This tool assists patients but does not replace professional medical advice.</small>
</center>
""", unsafe_allow_html=True)

