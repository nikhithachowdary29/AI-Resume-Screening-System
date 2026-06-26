# 📄 AI Resume Screening System

> An end-to-end Machine Learning application that classifies resumes into job categories, scores them for quality and ATS compatibility, extracts technical skills, and delivers actionable improvement suggestions — all through a clean Streamlit web interface.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Overview

The **AI Resume Screening System** is a final-year AI/ML portfolio project that automates the resume evaluation process. Recruiters and job seekers can upload a PDF resume or paste resume text to instantly receive:

- A predicted job category from **25 industry domains**
- A **Resume Score** (out of 100) measuring content quality
- An **ATS Compatibility Score** (out of 100) with a letter grade
- A categorised list of **350+ technical and soft skills** detected
- **Prioritised improvement suggestions** to strengthen the resume

The system is built on a **TF-IDF + Multinomial Naive Bayes** pipeline trained on real-world resume data, wrapped in a modular Python codebase and served through an interactive Streamlit dashboard.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📂 PDF Upload | Extract and analyse text directly from uploaded PDF resumes |
| 📝 Text Input | Paste raw resume text for instant analysis |
| 🎯 Job Category Prediction | Classifies resumes into 25 job roles with confidence scores |
| 📊 Confidence Breakdown | Top-8 category probabilities with visual progress bars |
| 🏆 Resume Score | Weighted 100-point score across skill breadth, depth, richness, and length |
| 🤖 ATS Score | ATS compatibility score with A–F grade and section-level feedback |
| 🛠️ Skill Extraction | Domain-grouped detection of 350+ technical and soft skills |
| 💡 Improvement Suggestions | Prioritised HIGH / MEDIUM / LOW actionable recommendations |
| 🎨 Professional UI | Dark-themed Streamlit dashboard with metric cards and expanders |
| ⚡ Modular Architecture | Cleanly separated training, utility, and application layers |

---

## 🏗️ Project Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      User Interface                     │
│                       (app.py)                          │
│         PDF Upload  ──  Paste Text  ──  Results         │
└──────────────────────────┬──────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
   ┌─────────────┐  ┌────────────┐  ┌────────────────┐
   │  ML Model   │  │ Skill      │  │ Score &        │
   │  Prediction │  │ Extraction │  │ Suggestion     │
   │  (joblib)   │  │ (utils.py) │  │ Engine         │
   └──────┬──────┘  └─────┬──────┘  └───────┬────────┘
          │               │                  │
          └───────────────▼──────────────────┘
                          │
              ┌───────────┴───────────┐
              │    Preprocessing      │
              │  fix_encoding()       │
              │  clean_text()         │
              │  TF-IDF Transform     │
              └───────────┬───────────┘
                          │
              ┌───────────┴───────────┐
              │   Trained Artifacts   │
              │  resume_model.joblib  │
              │  tfidf_vectorizer.joblib│
              │  label_encoder.joblib │
              └───────────────────────┘
```

**Training Pipeline** (`train_model.py`)

```
Raw CSV  →  Deduplication  →  Encoding Fix  →  Text Cleaning
       →  TF-IDF Vectorization  →  Train/Test Split (80/20)
       →  Multinomial Naive Bayes  →  Evaluation  →  Save Artifacts
```

---

## 🛠️ Technologies Used

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.9+ | Core development language |
| ML Framework | scikit-learn | TF-IDF, Naive Bayes, evaluation metrics |
| Model Persistence | Joblib | Serialize/deserialize trained artifacts |
| Data Processing | Pandas, NumPy | Dataset loading, array operations |
| PDF Parsing | pypdf | Extract text from PDF resumes |
| Web Interface | Streamlit | Interactive dashboard |
| Text Processing | re, string (stdlib) | Cleaning, encoding repair, pattern matching |

---

## 📁 Folder Structure

```
ai-resume-screening/
│
├── app.py                    # Streamlit web application
├── train_model.py            # ML training pipeline
├── utils.py                  # Utility functions (PDF, scoring, suggestions)
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
│
├── Resume_Screening.csv      # Training dataset (962 resumes, 25 categories)
│
└── models/                   # Auto-generated after training
    ├── resume_model.joblib       # Trained Multinomial Naive Bayes classifier
    ├── tfidf_vectorizer.joblib   # Fitted TF-IDF vectorizer
    └── label_encoder.joblib      # Fitted label encoder (int ↔ category name)
```

> **Note:** The `.joblib` files are generated by running `train_model.py` and are not committed to version control. Add them to `.gitignore`.

---

## ⚙️ Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git

### Steps

**1. Clone the repository**

```bash
git clone https://github.com/your-username/ai-resume-screening.git
cd ai-resume-screening
```

**2. Create and activate a virtual environment** *(recommended)*

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Place the dataset**

Ensure `Resume_Screening.csv` is in the project root directory.

---

## 📊 Dataset Information

| Property | Value |
|---|---|
| File | `Resume_Screening.csv` |
| Raw rows | 962 |
| Columns | `Category`, `Resume` |
| Unique categories | 25 |
| Unique resumes (post-dedup) | ~166 |
| Missing values | None |

### Job Categories (25)

```
Advocate          · Arts               · Automation Testing  · Blockchain
Business Analyst  · Civil Engineer     · Data Science        · Database
DevOps Engineer   · DotNet Developer   · ETL Developer       · Electrical Engineering
HR                · Hadoop             · Health and Fitness  · Java Developer
Mechanical Engg.  · Network Security   · Operations Manager  · PMO
Python Developer  · SAP Developer      · Sales               · Testing
Web Designing
```

### Data Quality Notes

- **Duplicates:** 796 out of 962 rows are exact duplicates (removed during preprocessing)
- **Encoding artifacts:** ~725 resumes contain UTF-8/Latin-1 mismatches (repaired via `fix_encoding()`)
- **Post-cleaning dataset:** 166 unique, clean resumes used for training

---

## 🤖 Model Training

Train the model by running:

```bash
python train_model.py
```

This will execute the full pipeline and print a detailed training log:

```
════════════════════════════════════════════════════════════
   AI RESUME SCREENING SYSTEM — MODEL TRAINING
════════════════════════════════════════════════════════════

  STEP 1: Loading Data           →  962 rows × 2 columns
  STEP 2: Preprocessing          →  166 unique resumes retained
  STEP 3: TF-IDF Vectorization   →  6,180 features (unigrams + bigrams)
  STEP 4: Train / Test Split     →  132 train / 34 test (stratified 80/20)
  STEP 5: Model Training         →  MultinomialNB (alpha=0.1)
  STEP 6: Evaluation             →  Accuracy: 76.47% · F1: 67.38%
  STEP 7: Save Artifacts         →  3 .joblib files written

  ✅ Training complete.
```

### Model Performance

| Metric | Score |
|---|---|
| Accuracy | 76.47% |
| Precision (macro) | 65.50% |
| Recall (macro) | 72.00% |
| F1 Score (macro) | 67.38% |

> **Note on scores:** The dataset contains only ~166 unique resumes across 25 categories (~6–7 per category after deduplication). Scores are honest reflections of the data size. A larger, deduplicated dataset would yield significantly higher performance. This project intentionally demonstrates robust preprocessing over artificially inflated metrics.

### Model Configuration

| Parameter | Value | Rationale |
|---|---|---|
| Algorithm | Multinomial Naive Bayes | Optimised for sparse TF-IDF text features |
| alpha (smoothing) | 0.1 | Reduced from default (1.0) for sharper class boundaries |
| TF-IDF max features | 15,000 | Caps vocabulary to reduce noise |
| N-gram range | (1, 2) | Captures phrases like "machine learning" |
| min_df | 2 | Removes hapax legomena (terms appearing once) |
| sublinear_tf | True | Log-dampens high-frequency terms |
| Train/test split | 80/20 | Stratified to preserve class proportions |

---

## 🚀 Running the Streamlit App

Ensure `train_model.py` has been run at least once before launching the app.

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

### Interface Overview

```
┌─────────────────────────────────────────────────┐
│  Sidebar              │  Main Panel             │
│  ─────────            │  ─────────              │
│  ✅ Model loaded      │  [Hero Banner]          │
│  ✅ Vectorizer loaded │                         │
│  ✅ Encoder loaded    │  [Tab: PDF Upload]      │
│  Categories: 25       │  [Tab: Paste Text]      │
│                       │                         │
│  How to Use           │  After analysis:        │
│  1. Upload/Paste      │  • Predicted Category   │
│  2. Click Analyse     │  • Confidence Chart     │
│  3. Review results    │  • Score Cards          │
│                       │  • ATS Details          │
│  Supported Categories │  • Extracted Skills     │
│  • Advocate           │  • Suggestions          │
│  • Arts ...           │                         │
└─────────────────────────────────────────────────┘
```

---

## 🔄 Example Workflow

**1. Upload a PDF resume**

Navigate to the **Upload PDF Resume** tab, drag and drop a PDF file, and click **Analyse Resume**.

**2. Review predicted category**

```
💼  Python Developer          Confidence: 91.3%
████████████████████████████████████░░░░  91.3%
```

**3. Check scores**

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Resume Score │  │  ATS Score   │  │ Skills Found │  │  Word Count  │
│    72/100    │  │   85/100     │  │    24/100    │  │   487/100    │
│   Average    │  │  Grade: A    │  │  6 domains   │  │ Ideal: 300–9 │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
```

**4. Explore extracted skills by domain**

```
Programming Languages (3)   python  · javascript  · typescript
Data Science & ML (7)       machine learning  · scikit-learn  · pandas  · ...
Cloud & DevOps (4)          docker  · kubernetes  · aws  · github actions
```

**5. Act on improvement suggestions**

```
🔴 HIGH   · Contact Info     Add a phone number. Recruiters expect direct contact.
🔴 HIGH   · Structure        Add a clearly labelled 'Work Experience' section.
🟡 MEDIUM · Role Alignment   For 'Python Developer', consider adding: fastapi, pytest.
🟢 LOW    · Impact           Add quantified achievements (e.g. 'Reduced load time 40%').
```

---

## 🔮 Future Enhancements

- [ ] **Larger dataset** — augment with LinkedIn, Kaggle, or proprietary resume corpora to improve per-category accuracy
- [ ] **Deep learning classifier** — replace Naive Bayes with a fine-tuned BERT or DistilBERT model for contextual understanding
- [ ] **Job description matching** — paste a JD alongside the resume to compute a tailored match score
- [ ] **Multi-language support** — extend preprocessing and skill detection to non-English resumes
- [ ] **OCR for scanned PDFs** — integrate Tesseract / EasyOCR to handle image-based PDF resumes
- [ ] **Resume anonymisation** — redact PII before analysis to enable fair, bias-aware screening
- [ ] **Batch processing** — upload and analyse multiple resumes simultaneously with CSV export
- [ ] **REST API** — expose prediction and scoring endpoints via FastAPI for third-party integration
- [ ] **User authentication** — add login and history tracking for repeat users
- [ ] **Docker deployment** — containerise the app for one-command cloud deployment

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

## 👤 Author

**Your Name**
Final Year B.Tech — Artificial Intelligence & Machine Learning

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin&logoColor=white)](https://linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?logo=github&logoColor=white)](https://github.com/your-username)
[![Email](https://img.shields.io/badge/Email-Contact-D14836?logo=gmail&logoColor=white)](mailto:your.email@example.com)

---

## ⭐ Acknowledgements

- Dataset sourced from [Kaggle — Resume Dataset](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset)
- Built with [Streamlit](https://streamlit.io) · [scikit-learn](https://scikit-learn.org) · [pypdf](https://github.com/py-pdf/pypdf)

---

*If this project helped you, please consider giving it a ⭐ on GitHub!*
