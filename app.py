# app.py
# Main Streamlit application for the Mini Resume Analyzer (Skill Gap Detector).
# Run with: streamlit run app.py

import streamlit as st
import spacy
import plotly.graph_objects as go

# Import our custom modules
from utils import extract_text_from_pdf, preprocess_text, truncate_text
from skill_extractor import extract_skills, load_spacy_model
from similarity import compute_match_percentage, get_skill_gap_summary

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Mini Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS for better styling
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .skill-tag-green {
        display: inline-block;
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
        border-radius: 15px;
        padding: 4px 12px;
        margin: 3px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .skill-tag-red {
        display: inline-block;
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
        border-radius: 15px;
        padding: 4px 12px;
        margin: 3px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .skill-tag-blue {
        display: inline-block;
        background-color: #cce5ff;
        color: #004085;
        border: 1px solid #b8daff;
        border-radius: 15px;
        padding: 4px 12px;
        margin: 3px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    .section-divider {
        border-top: 2px solid #e9ecef;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Load spaCy model (cached so it loads only once)
# ─────────────────────────────────────────────


@st.cache_resource
def load_model():
    return spacy.load("en_core_web_sm")

nlp = load_model()


# ─────────────────────────────────────────────
# Header Section
# ─────────────────────────────────────────────
st.markdown('<div class="main-header">📄 Mini Resume Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload your resume & paste a job description to discover your skill gaps</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar — How to Use
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ How to Use")
    st.markdown("""
    1. **Upload** your resume as a PDF file
    2. **Paste** the job description in the text box
    3. Click **Analyze** to see your results
    4. Review:
       - ✅ Matching skills
       - ❌ Missing skills
       - 📊 Match percentage
    """)
    st.divider()
    st.header("🔧 About")
    st.markdown("""
    This tool uses:
    - **pdfplumber** for PDF parsing
    - **spaCy** for NLP processing
    - **scikit-learn** for similarity scoring
    - **Streamlit** for the UI
    """)
    st.divider()
    st.caption("Built with ❤️ using Python")


# ─────────────────────────────────────────────
# Main Input Section — Two Columns
# ─────────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("📎 Upload Resume (PDF)")
    uploaded_file = st.file_uploader(
        label="Choose your resume PDF",
        type=["pdf"],
        help="Only PDF files are supported.",
    )
    # Preview extracted text in an expander
    if uploaded_file:
        with st.spinner("Reading PDF..."):
            raw_resume_text = extract_text_from_pdf(uploaded_file)
            resume_text = preprocess_text(raw_resume_text)

        if resume_text:
            st.success(f"✅ PDF loaded! ({len(resume_text)} characters extracted)")
            with st.expander("👁️ Preview Extracted Resume Text"):
                st.text_area(
                    label="Extracted Text",
                    value=truncate_text(resume_text, 2000),
                    height=200,
                    disabled=True,
                )
        else:
            st.error("❌ Could not extract text from the PDF. Try another file.")
            resume_text = ""
    else:
        resume_text = ""

with col2:
    st.subheader("📋 Paste Job Description")
    jd_text = st.text_area(
        label="Job Description",
        placeholder="Paste the full job description here...\n\nExample:\nWe are looking for a Python developer with experience in Django, REST APIs, PostgreSQL, Docker, and AWS...",
        height=250,
        help="Paste the complete job description for best results.",
    )

# ─────────────────────────────────────────────
# Analyze Button
# ─────────────────────────────────────────────
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

analyze_button = st.button(
    label="🔍 Analyze Resume",
    type="primary",
    use_container_width=True,
    disabled=(not uploaded_file or not jd_text.strip()),
)

# Show hint if inputs are missing
if not uploaded_file or not jd_text.strip():
    st.info("👆 Please upload a resume PDF and paste a job description to enable analysis.")

# ─────────────────────────────────────────────
# Analysis Results
# ─────────────────────────────────────────────
if analyze_button and resume_text and jd_text.strip():
    nlp = get_nlp_model()

    with st.spinner("🔄 Analyzing your resume... Please wait."):

        # Step 1: Extract skills from resume and job description
        resume_skills = extract_skills(resume_text)
        jd_skills = extract_skills(jd_text)

        # Step 2: Compute text similarity (TF-IDF cosine similarity)
        match_percentage = compute_match_percentage(resume_text, jd_text)

        # Step 3: Get skill gap summary
        summary = get_skill_gap_summary(resume_skills, jd_skills)

    # ── Display Results ────────────────────────
    st.success("✅ Analysis Complete!")
    st.markdown("---")
    st.header("📊 Analysis Dashboard")

    # ── Top Metrics Row ───────────────────────
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(
            label="🎯 Overall Match",
            value=f"{match_percentage}%",
            help="Cosine similarity between resume and job description text.",
        )
    with m2:
        st.metric(
            label="✅ Skills Matched",
            value=summary["matched_count"],
            help="Skills found in both resume and job description.",
        )
    with m3:
        st.metric(
            label="❌ Skills Missing",
            value=summary["missing_count"],
            help="Skills required by JD but absent from resume.",
        )
    with m4:
        st.metric(
            label="📋 Skill Coverage",
            value=f"{summary['skill_coverage_percentage']}%",
            help="Percentage of JD skills that appear in your resume.",
        )

    st.markdown("---")

    # ── Donut Chart ───────────────────────────
    st.subheader("🍩 Skill Match Breakdown")

    chart_col, info_col = st.columns([1, 1], gap="large")

    with chart_col:
        matched = summary["matched_count"]
        missing = summary["missing_count"]
        total = matched + missing

        if total > 0:
            fig = go.Figure(data=[go.Pie(
                labels=["Matched Skills", "Missing Skills"],
                values=[matched, missing],
                hole=0.5,
                marker=dict(colors=["#28a745", "#dc3545"]),
                textinfo="label+percent",
                hoverinfo="label+value",
            )])
            fig.update_layout(
                showlegend=True,
                height=300,
                margin=dict(t=10, b=10, l=10, r=10),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No skills detected to chart.")

    with info_col:
        # Match score gauge
        st.markdown("**Resume Match Score**")
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=match_percentage,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1f77b4"},
                "steps": [
                    {"range": [0, 40], "color": "#f8d7da"},
                    {"range": [40, 70], "color": "#fff3cd"},
                    {"range": [70, 100], "color": "#d4edda"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": match_percentage,
                },
            },
        ))
        gauge_fig.update_layout(height=250, margin=dict(t=20, b=10, l=30, r=30))
        st.plotly_chart(gauge_fig, use_container_width=True)

    st.markdown("---")

    # ── Skills Detail Section ─────────────────
    tab1, tab2, tab3 = st.tabs(["✅ Matching Skills", "❌ Missing Skills", "📄 All Extracted Skills"])

    with tab1:
        st.subheader("✅ Skills You Already Have")
        if summary["matching_skills"]:
            # Display each skill as a colored tag
            tags_html = "".join(
                f'<span class="skill-tag-green">{skill}</span>'
                for skill in summary["matching_skills"]
            )
            st.markdown(tags_html, unsafe_allow_html=True)
            st.caption(f"Total: {len(summary['matching_skills'])} matching skill(s)")
        else:
            st.warning("No matching skills found. Consider tailoring your resume to the job description.")

    with tab2:
        st.subheader("❌ Skills You Need to Add / Develop")
        if summary["missing_skills"]:
            tags_html = "".join(
                f'<span class="skill-tag-red">{skill}</span>'
                for skill in summary["missing_skills"]
            )
            st.markdown(tags_html, unsafe_allow_html=True)
            st.caption(f"Total: {len(summary['missing_skills'])} missing skill(s)")

            # Pro tip section
            st.markdown("---")
            st.info("💡 **Pro Tip:** Focus on the top 3-5 missing skills that are most relevant to the role. "
                    "Consider taking online courses, building projects, or adding these to your resume if you have experience.")
        else:
            st.success("🎉 Great news! Your resume covers all detected skills from the job description.")

    with tab3:
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("📄 Resume Skills")
            if resume_skills:
                tags_html = "".join(
                    f'<span class="skill-tag-blue">{skill}</span>'
                    for skill in resume_skills
                )
                st.markdown(tags_html, unsafe_allow_html=True)
                st.caption(f"Total: {len(resume_skills)} skill(s) detected")
            else:
                st.warning("No skills detected in resume. Make sure skills are clearly listed.")

        with col_b:
            st.subheader("📋 Job Description Skills")
            if jd_skills:
                tags_html = "".join(
                    f'<span class="skill-tag-blue">{skill}</span>'
                    for skill in jd_skills
                )
                st.markdown(tags_html, unsafe_allow_html=True)
                st.caption(f"Total: {len(jd_skills)} skill(s) detected")
            else:
                st.warning("No skills detected in job description.")

    st.markdown("---")

    # ── Recommendation Section ────────────────
    st.subheader("🚀 Personalized Recommendations")

    score = summary["skill_coverage_percentage"]
    if score >= 80:
        st.success("🌟 **Excellent Match!** Your profile aligns very well with this role. Focus on tailoring your experience bullet points to the job description.")
    elif score >= 60:
        st.warning("👍 **Good Match.** You're a solid candidate. Work on bridging a few skill gaps to strengthen your application.")
    elif score >= 40:
        st.warning("⚠️ **Moderate Match.** You meet some requirements but should upskill in key areas before applying.")
    else:
        st.error("❌ **Low Match.** This role may need significant upskilling. Consider building the missing skills before applying.")
