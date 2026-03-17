# 📄 Mini Resume Analyzer — Skill Gap Detector

A beginner-friendly Python app that compares your resume against a job description to identify matching and missing skills, powered by NLP and machine learning.

---

## 🚀 Features

| Feature | Description |
|---|---|
| 📎 PDF Upload | Upload your resume as a PDF file |
| 🔍 Skill Extraction | Automatically extract skills from resume & JD |
| 📊 Match Score | TF-IDF cosine similarity score |
| ✅ Matching Skills | Skills you already have that match the JD |
| ❌ Missing Skills | Skills required by the JD that you need |
| 🍩 Visual Dashboard | Charts, gauges, and colored skill tags |
| 💡 Recommendations | Personalized tips based on your match score |

---

## 🗂️ Project Structure

```
resume-analyzer/
├── app.py              # Main Streamlit UI application
├── skill_extractor.py  # NLP-based skill extraction logic
├── similarity.py       # Skill comparison & match scoring
├── utils.py            # PDF parsing and text utilities
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## ⚙️ Setup & Installation

### 1. Clone or download the project

```bash
git clone https://github.com/yourname/resume-analyzer.git
cd resume-analyzer
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the spaCy language model

```bash
python -m spacy download en_core_web_sm
```

### 5. Run the app

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

---

## 🎯 How to Use

1. **Upload your resume** — Click "Browse files" and select your PDF resume
2. **Paste the job description** — Copy the full JD text into the text area
3. **Click "Analyze Resume"** — The system will process both inputs
4. **Review your results:**
   - 🎯 **Overall Match %** — Text similarity score
   - ✅ **Matching Skills** — Skills you already have
   - ❌ **Missing Skills** — Skills you need to develop
   - 📊 **Skill Coverage %** — How many JD skills you cover

---

## 🧠 How It Works

### Skill Extraction (`skill_extractor.py`)
- Maintains a curated database of 80+ tech and soft skills
- Uses regex with word boundary matching to find skills in text
- Normalizes text to lowercase before matching to avoid case issues

### Similarity Scoring (`similarity.py`)
- Converts resume and JD text into TF-IDF vectors
- Computes cosine similarity to produce a match percentage
- Calculates skill-level gap analysis (matched vs. missing)

### PDF Parsing (`utils.py`)
- Uses `pdfplumber` to extract text from all pages of a PDF
- Cleans and preprocesses extracted text for analysis

---

## 📦 Tech Stack

| Library | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `pdfplumber` | PDF text extraction |
| `spacy` | NLP processing |
| `scikit-learn` | TF-IDF & cosine similarity |
| `plotly` | Interactive charts |

---

## 🔧 Customizing the Skill Database

To add more skills, open `skill_extractor.py` and add entries to the `SKILLS_DB` list:

```python
SKILLS_DB = [
    # Add your custom skills here
    "your new skill",
    "another skill",
    ...
]
```

---

## 📌 Tips for Best Results

- **Resume:** Use a clean, text-based PDF (not a scanned image)
- **Job Description:** Paste the full JD including requirements section
- **Skills:** Make sure your resume explicitly lists your skills (e.g., a "Skills" section)

---

## 📄 License

MIT License — free to use, modify, and distribute.
