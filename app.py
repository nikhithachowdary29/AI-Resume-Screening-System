"""
app.py
======
AI Resume Screening System — Streamlit Interface

Provides a clean, professional web UI for:
  - Uploading PDF resumes or pasting plain text
  - Predicting job category with confidence breakdown
  - Displaying resume score, ATS score, extracted skills
  - Showing prioritised improvement suggestions

Run with:
    streamlit run app.py

Dependencies:
    pip install streamlit pypdf scikit-learn joblib pandas
"""

import io
import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import streamlit as st

from utils import (
    extract_text_from_pdf,
    clean_text,
    extract_skills,
    get_all_detected_skills,
    calculate_resume_score,
    calculate_ats_score,
    generate_suggestions,
    SKILLS_REGISTRY,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Global ── */
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

/* ── Header banner ── */
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem 2rem;
    margin-bottom: 1.8rem;
    text-align: center;
}
.hero h1 { color: #ffffff; font-size: 2.2rem; font-weight: 700; margin: 0 0 .4rem; }
.hero p  { color: #a8b2d8; font-size: 1rem; margin: 0; }

/* ── Score cards ── */
.score-card {
    background: #0f3460;
    border-radius: 14px;
    padding: 1.4rem 1rem;
    text-align: center;
    color: #ffffff;
    box-shadow: 0 4px 20px rgba(0,0,0,.25);
}
.score-card .label { font-size: .78rem; color: #a8b2d8; text-transform: uppercase;
                     letter-spacing: .08em; margin-bottom: .3rem; }
.score-card .value { font-size: 2.6rem; font-weight: 800; line-height: 1; }
.score-card .sub   { font-size: .8rem;  color: #a8b2d8; margin-top: .25rem; }

/* ── Category badge ── */
.category-badge {
    display: inline-block;
    background: linear-gradient(90deg, #e94560, #0f3460);
    color: #ffffff;
    border-radius: 24px;
    padding: .45rem 1.4rem;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: .03em;
    margin: .5rem 0 1rem;
}

/* ── Skill pill ── */
.skill-pill {
    display: inline-block;
    background: #16213e;
    color: #e2e8f0;
    border: 1px solid #0f3460;
    border-radius: 20px;
    padding: .28rem .85rem;
    font-size: .78rem;
    margin: .2rem .18rem;
    white-space: nowrap;
}
.skill-pill.highlight { background: #0f3460; border-color: #e94560; color: #fff; }

/* ── Suggestion rows ── */
.sug-high   { border-left: 4px solid #e94560; background: #1a1230;
              border-radius: 8px; padding: .7rem 1rem; margin: .45rem 0; }
.sug-medium { border-left: 4px solid #f59e0b; background: #1a160a;
              border-radius: 8px; padding: .7rem 1rem; margin: .45rem 0; }
.sug-low    { border-left: 4px solid #10b981; background: #0a1a12;
              border-radius: 8px; padding: .7rem 1rem; margin: .45rem 0; }
.sug-label  { font-size: .7rem; font-weight: 700; text-transform: uppercase;
              letter-spacing: .07em; margin-bottom: .2rem; }
.sug-high   .sug-label { color: #e94560; }
.sug-medium .sug-label { color: #f59e0b; }
.sug-low    .sug-label { color: #10b981; }
.sug-cat    { color: #a8b2d8; font-size: .72rem; }
.sug-msg    { color: #e2e8f0; font-size: .88rem; margin-top: .15rem; }

/* ── Section headings ── */
.section-title {
    font-size: 1.05rem; font-weight: 700; color: #e2e8f0;
    border-bottom: 2px solid #0f3460;
    padding-bottom: .4rem; margin: 1.2rem 0 .8rem;
}

/* ── ATS grade ── */
.ats-grade {
    font-size: 2.2rem; font-weight: 900;
    background: linear-gradient(135deg, #e94560, #f59e0b);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}

/* ── Confidence bar container ── */
.conf-row { display: flex; align-items: center; gap: .6rem; margin: .3rem 0; }
.conf-name { width: 190px; font-size: .82rem; color: #a8b2d8;
             white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.conf-pct  { width: 46px; text-align: right; font-size: .82rem;
             color: #e2e8f0; font-weight: 600; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: #0d0d1a; }

/* ── Upload area ── */
[data-testid="stFileUploader"] { border-radius: 12px; }

/* ── Expander tweaks ── */
details summary { font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING  (cached — loaded once per session)
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH      = "resume_model.joblib"
VECTORIZER_PATH = "tfidf_vectorizer.joblib"
ENCODER_PATH    = "label_encoder.joblib"


@st.cache_resource(show_spinner=False)
def load_model_artifacts():
    """Load and cache the trained ML artifacts from disk."""
    missing = [p for p in [MODEL_PATH, VECTORIZER_PATH, ENCODER_PATH] if not os.path.exists(p)]
    if missing:
        return None, None, None, missing
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    encoder    = joblib.load(ENCODER_PATH)
    return model, vectorizer, encoder, []


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def predict_category(raw_text: str, model, vectorizer, encoder) -> tuple[str, float, dict]:
    """
    Predict the job category for a resume and return confidence scores.

    Returns
    -------
    predicted_label : str
    top_confidence  : float   (0–1)
    top_confidences : dict    {label: confidence} for top-10 categories
    """
    cleaned = clean_text(raw_text)
    X = vectorizer.transform([cleaned])
    proba = model.predict_proba(X)[0]
    predicted_idx = int(np.argmax(proba))
    predicted_label = encoder.classes_[predicted_idx]
    top_confidence = float(proba[predicted_idx])

    # Top-10 categories by confidence
    top_indices = np.argsort(proba)[::-1][:10]
    top_confidences = {
        encoder.classes_[i]: float(proba[i])
        for i in top_indices
    }
    return predicted_label, top_confidence, top_confidences


# ─────────────────────────────────────────────────────────────────────────────
# UI RENDERING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def render_score_card(label: str, value: int, sub: str = "") -> str:
    return f"""
    <div class="score-card">
        <div class="label">{label}</div>
        <div class="value">{value}<span style="font-size:1.1rem;opacity:.6">/100</span></div>
        {"<div class='sub'>" + sub + "</div>" if sub else ""}
    </div>"""


def score_color(score: int) -> str:
    if score >= 75: return "#10b981"
    if score >= 50: return "#f59e0b"
    return "#e94560"


def render_skill_pills(skills: list[str], highlight: list[str] = None) -> str:
    highlight = set(highlight or [])
    pills = ""
    for skill in sorted(skills):
        cls = "skill-pill highlight" if skill in highlight else "skill-pill"
        pills += f'<span class="{cls}">{skill}</span>'
    return pills


def render_suggestion(sug: dict) -> str:
    p = sug["priority"]
    return f"""
    <div class="sug-{p}">
        <div class="sug-label">{'🔴' if p=='high' else '🟡' if p=='medium' else '🟢'} {p} priority
            <span class="sug-cat"> · {sug['category']}</span>
        </div>
        <div class="sug-msg">{sug['message']}</div>
    </div>"""


def render_confidence_bars(top_confidences: dict, predicted: str) -> None:
    """Render horizontal confidence bars using Streamlit progress."""
    for label, conf in list(top_confidences.items())[:8]:
        col1, col2 = st.columns([3, 1])
        with col1:
            is_top = label == predicted
            display = f"**{label}**" if is_top else label
            st.markdown(display)
            st.progress(conf)
        with col2:
            st.markdown(f"<div style='padding-top:.55rem;color:#a8b2d8;font-size:.85rem'>{conf*100:.1f}%</div>",
                        unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ANALYSIS FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(raw_text: str, model, vectorizer, encoder) -> None:
    """Execute full analysis pipeline and render all result sections."""

    if len(raw_text.strip()) < 50:
        st.error("⚠️ Resume text is too short. Please upload a valid resume or paste more content.")
        return

    with st.spinner("🔍 Analysing resume..."):

        # ── ML Prediction ──────────────────────────────────────────────────
        try:
            predicted_category, top_confidence, top_confidences = predict_category(
                raw_text, model, vectorizer, encoder
            )
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return

        # ── Utils pipeline ─────────────────────────────────────────────────
        try:
            extracted_skills = extract_skills(raw_text)
            all_skills       = get_all_detected_skills(extracted_skills)
            resume_score_res = calculate_resume_score(raw_text, extracted_skills, predicted_category)
            ats_res          = calculate_ats_score(raw_text, extracted_skills)
            suggestions      = generate_suggestions(
                raw_text, resume_score_res, ats_res, extracted_skills, predicted_category
            )
        except Exception as e:
            st.error(f"Analysis error: {e}")
            return

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — PREDICTED CATEGORY
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 🎯 Predicted Job Category")

    col_cat, col_conf = st.columns([1, 1])

    with col_cat:
        st.markdown(
            f'<div class="category-badge">💼 {predicted_category}</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Model confidence: **{top_confidence*100:.1f}%**")
        conf_color = score_color(int(top_confidence * 100))
        st.progress(top_confidence)

    with col_conf:
        with st.expander("📊 View confidence for all categories", expanded=False):
            render_confidence_bars(top_confidences, predicted_category)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — SCORE CARDS
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 📈 Resume Analysis Scores")

    total   = resume_score_res["total_score"]
    ats     = ats_res["ats_score"]
    skills_count = resume_score_res["total_skills"]
    domains = resume_score_res["domains_covered"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(render_score_card("Resume Score", total,
            "Good" if total >= 75 else "Average" if total >= 50 else "Needs Work"),
            unsafe_allow_html=True)
    with c2:
        st.markdown(render_score_card("ATS Score", ats,
            f"Grade: {ats_res['ats_grade']}"),
            unsafe_allow_html=True)
    with c3:
        st.markdown(render_score_card("Skills Found", skills_count,
            f"Across {domains} domains"),
            unsafe_allow_html=True)
    with c4:
        word_count = resume_score_res.get("word_count", 0)
        st.markdown(render_score_card("Word Count", word_count,
            "Ideal: 300–900"),
            unsafe_allow_html=True)

    # Score sub-breakdowns
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📋 Score Breakdown", expanded=True):

        sub1, sub2 = st.columns(2)

        with sub1:
            st.markdown("**Resume Score Components**")
            dims = {
                "Skill Breadth":   (resume_score_res["skill_score"],   40),
                "Domain Coverage": (resume_score_res["domain_score"],  25),
                "Content Richness":(resume_score_res["richness_score"],20),
                "Length & Detail": (resume_score_res["length_score"],  15),
            }
            for dim_name, (score, max_score) in dims.items():
                pct = score / max_score
                col_l, col_r = st.columns([3, 1])
                col_l.markdown(f"<small>{dim_name}</small>", unsafe_allow_html=True)
                col_r.markdown(
                    f"<small style='color:#a8b2d8'>{score}/{max_score}</small>",
                    unsafe_allow_html=True)
                st.progress(pct)

        with sub2:
            st.markdown("**ATS Score Components**")
            ats_dims = {
                "Standard Sections": (ats_res["sections_score"],   35),
                "Keyword Density":   (ats_res["keyword_score"],    30),
                "Contact Info":      (ats_res["contact_score"],    20),
                "Formatting":        (ats_res["formatting_score"], 15),
            }
            for dim_name, (score, max_score) in ats_dims.items():
                pct = score / max_score
                col_l, col_r = st.columns([3, 1])
                col_l.markdown(f"<small>{dim_name}</small>", unsafe_allow_html=True)
                col_r.markdown(
                    f"<small style='color:#a8b2d8'>{score}/{max_score}</small>",
                    unsafe_allow_html=True)
                st.progress(pct)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — ATS DETAILS
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 🤖 ATS Compatibility Details")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown(
            f'<div style="text-align:center"><div class="ats-grade">{ats_res["ats_grade"]}</div>'
            f'<div style="color:#a8b2d8;font-size:.85rem">ATS Grade</div></div>',
            unsafe_allow_html=True,
        )

    with col_b:
        found = ats_res.get("sections_found", [])
        st.markdown("**✅ Sections Detected**")
        if found:
            for s in found:
                st.markdown(f"<small>✓ {s.title()}</small>", unsafe_allow_html=True)
        else:
            st.caption("None detected")

    with col_c:
        missing = ats_res.get("sections_missing", [])
        st.markdown("**❌ Sections Missing**")
        if missing:
            for s in missing:
                st.markdown(f"<small style='color:#e94560'>✗ {s.title()}</small>",
                            unsafe_allow_html=True)
        else:
            st.success("All key sections present!")

    # Contact completeness
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Contact Information Detected**")
    contact = ats_res.get("contact_details", {})
    c_cols = st.columns(4)
    icons = {"email": "📧 Email", "phone": "📞 Phone", "linkedin": "🔗 LinkedIn", "location": "📍 Location"}
    for i, (key, label) in enumerate(icons.items()):
        present = contact.get(key, False)
        color = "#10b981" if present else "#e94560"
        mark  = "✓" if present else "✗"
        c_cols[i].markdown(
            f"<div style='text-align:center;color:{color};font-size:.9rem'>"
            f"{mark} {label}</div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — EXTRACTED SKILLS
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 🛠️ Extracted Skills")

    if all_skills:
        st.markdown(
            f"**{len(all_skills)} skills detected** across **{domains} domains**",
        )

        # Domain-grouped pills
        for domain, domain_skills in sorted(extracted_skills.items()):
            domain_label = domain.replace("_", " ").title()
            with st.expander(f"**{domain_label}** — {len(domain_skills)} skills", expanded=True):
                st.markdown(
                    render_skill_pills(domain_skills, highlight=domain_skills[:3]),
                    unsafe_allow_html=True,
                )
    else:
        st.warning("No skills were detected. The resume may be image-based or have unusual formatting.")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5 — IMPROVEMENT SUGGESTIONS
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 💡 Resume Improvement Suggestions")

    if suggestions:
        high   = [s for s in suggestions if s["priority"] == "high"]
        medium = [s for s in suggestions if s["priority"] == "medium"]
        low    = [s for s in suggestions if s["priority"] == "low"]

        # Summary counts
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("🔴 High Priority",   len(high))
        sc2.metric("🟡 Medium Priority", len(medium))
        sc3.metric("🟢 Low Priority",    len(low))

        st.markdown("<br>", unsafe_allow_html=True)

        if high:
            with st.expander(f"🔴 High Priority ({len(high)})", expanded=True):
                for sug in high:
                    st.markdown(render_suggestion(sug), unsafe_allow_html=True)

        if medium:
            with st.expander(f"🟡 Medium Priority ({len(medium)})", expanded=True):
                for sug in medium:
                    st.markdown(render_suggestion(sug), unsafe_allow_html=True)

        if low:
            with st.expander(f"🟢 Low Priority ({len(low)})", expanded=False):
                for sug in low:
                    st.markdown(render_suggestion(sug), unsafe_allow_html=True)
    else:
        st.success("✅ No major issues found! This resume looks strong.")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6 — RAW CLEAN TEXT (collapsible)
    # ══════════════════════════════════════════════════════════════════════════
    with st.expander("🔍 View Cleaned Resume Text (used for ML prediction)", expanded=False):
        st.code(clean_text(raw_text), language=None)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar(model, vectorizer, encoder, missing_files: list) -> None:
    with st.sidebar:
        st.markdown("## ⚙️ System Status")

        if missing_files:
            st.error(f"Missing model files:\n" + "\n".join(f"- `{f}`" for f in missing_files))
            st.info("Run `python train_model.py` to generate the model artifacts.")
        else:
            st.success("✅ Model loaded")
            st.success("✅ Vectorizer loaded")
            st.success("✅ Label encoder loaded")
            if encoder is not None:
                st.metric("Job Categories", len(encoder.classes_))

        st.markdown("---")
        st.markdown("## 📖 How to Use")
        st.markdown("""
1. **Upload** a PDF resume  
   *or*  
   **Paste** resume text  

2. Click **Analyse Resume**

3. Review:
   - Predicted job category
   - Resume & ATS scores
   - Extracted skills
   - Improvement suggestions
""")
        st.markdown("---")
        st.markdown("## 🗂️ Supported Categories")
        if encoder is not None:
            for cat in sorted(encoder.classes_):
                st.markdown(f"<small>• {cat}</small>", unsafe_allow_html=True)

        st.markdown("---")
        st.caption("AI Resume Screening System · Built with Streamlit")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:

    # Load model artifacts
    model, vectorizer, encoder, missing_files = load_model_artifacts()

    # Sidebar
    render_sidebar(model, vectorizer, encoder, missing_files)

    # ── Hero banner ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <h1>📄 AI Resume Screening System</h1>
        <p>Upload or paste a resume · Predict job category · Get scores & actionable feedback</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Hard stop if model not available ────────────────────────────────────
    if missing_files:
        st.error(
            "**Model artifacts not found.**\n\n"
            f"Missing: {', '.join(f'`{f}`' for f in missing_files)}\n\n"
            "Please run `python train_model.py` first to generate the model, "
            "then restart this app."
        )
        st.stop()

    # ── Input tabs ───────────────────────────────────────────────────────────
    tab_pdf, tab_text = st.tabs(["📎 Upload PDF Resume", "📝 Paste Resume Text"])

    raw_text: str = ""

    with tab_pdf:
        st.markdown("#### Upload your resume as a PDF file")
        uploaded_file = st.file_uploader(
            "Drag and drop or click to browse",
            type=["pdf"],
            help="Only text-based PDFs are supported. Scanned image PDFs may not extract correctly.",
        )

        if uploaded_file is not None:
            file_details_col, _ = st.columns([2, 1])
            with file_details_col:
                st.info(
                    f"📄 **{uploaded_file.name}** · "
                    f"{uploaded_file.size / 1024:.1f} KB"
                )

            with st.spinner("Extracting text from PDF..."):
                try:
                    raw_text = extract_text_from_pdf(uploaded_file)
                except RuntimeError as e:
                    st.error(f"PDF library error: {e}")
                    raw_text = ""
                except Exception as e:
                    st.error(f"Could not read PDF: {e}")
                    raw_text = ""

            if raw_text.startswith("[WARNING"):
                st.warning(raw_text)
                raw_text = ""
            elif raw_text.strip():
                word_count = len(raw_text.split())
                st.success(f"✅ Text extracted successfully — **{word_count} words** detected.")

                with st.expander("👁️ Preview extracted text", expanded=False):
                    preview = raw_text[:1500] + ("…" if len(raw_text) > 1500 else "")
                    st.text_area("Extracted text", preview, height=250, disabled=True)
            else:
                st.warning(
                    "No text could be extracted from this PDF. "
                    "It may be a scanned image. Try copying the text and using the **Paste Resume Text** tab."
                )

        if raw_text.strip():
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔍 Analyse Resume", type="primary", use_container_width=True, key="btn_pdf"):
                run_analysis(raw_text, model, vectorizer, encoder)

    with tab_text:
        st.markdown("#### Paste your resume text below")
        pasted = st.text_area(
            "Resume content",
            height=320,
            placeholder=(
                "John Doe | john@email.com | linkedin.com/in/johndoe\n\n"
                "SUMMARY\nExperienced Python Developer with 4 years ...\n\n"
                "SKILLS\nPython, Django, Flask, PostgreSQL, Docker, AWS ...\n\n"
                "EXPERIENCE\nSoftware Engineer - Acme Corp (2021–2025)\n..."
            ),
            help="Paste the full resume text including all sections.",
        )

        char_count = len(pasted.strip())
        if char_count > 0:
            st.caption(f"{char_count} characters · {len(pasted.split())} words")

        if pasted.strip():
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔍 Analyse Resume", type="primary", use_container_width=True, key="btn_text"):
                run_analysis(pasted, model, vectorizer, encoder)
        else:
            st.info("💡 Paste your resume text above, then click **Analyse Resume**.")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
