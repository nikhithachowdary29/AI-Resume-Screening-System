"""
utils.py
========
AI Resume Screening System — Utility Functions

Provides modular helpers for:
  - PDF text extraction (via pypdf, with PyPDF2 compatibility shim)
  - Text cleaning & preprocessing (mirrors train_model.py pipeline exactly)
  - Technical skill extraction from a configurable skills registry
  - Resume scoring (out of 100) based on skill coverage
  - ATS compatibility scoring
  - Resume improvement suggestions

All functions are stateless and importable independently.
"""

import re
import io
import os
import string
from typing import Optional

# ── PDF backend: prefer pypdf (modern); shim PyPDF2 name for callers ──────────
try:
    import pypdf as PyPDF2                         # pypdf 3.x / 5.x API
    _PDF_BACKEND = "pypdf"
except ImportError:
    try:
        import PyPDF2                              # legacy PyPDF2
        _PDF_BACKEND = "PyPDF2"
    except ImportError:
        PyPDF2 = None
        _PDF_BACKEND = None


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURABLE SKILLS REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
# Each top-level key is a domain; values are lists of skills/keywords.
# Extend or override this dict to customise extraction for any job category.

SKILLS_REGISTRY: dict[str, list[str]] = {
    "programming_languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "c",
        "r", "go", "golang", "ruby", "php", "swift", "kotlin", "scala",
        "rust", "perl", "matlab", "bash", "shell", "powershell", "vba",
        "groovy", "lua", "dart", "elixir", "haskell", "f#",
    ],
    "web_technologies": [
        "html", "css", "html5", "css3", "react", "angular", "vue",
        "node", "nodejs", "express", "django", "flask", "fastapi",
        "spring", "springboot", "asp.net", "jquery", "bootstrap",
        "sass", "less", "webpack", "graphql", "rest", "restful", "soap",
        "next.js", "nuxt", "gatsby", "redux", "tailwind",
    ],
    "data_science_ml": [
        "machine learning", "deep learning", "neural network", "nlp",
        "natural language processing", "computer vision", "tensorflow",
        "keras", "pytorch", "scikit-learn", "sklearn", "xgboost",
        "lightgbm", "pandas", "numpy", "scipy", "matplotlib", "seaborn",
        "plotly", "tableau", "power bi", "statistics", "regression",
        "classification", "clustering", "random forest", "svm",
        "reinforcement learning", "bert", "transformer", "llm",
        "feature engineering", "data wrangling", "eda",
    ],
    "databases": [
        "sql", "mysql", "postgresql", "postgres", "sqlite", "oracle",
        "mongodb", "cassandra", "redis", "elasticsearch", "dynamodb",
        "hbase", "neo4j", "mariadb", "ms sql", "sql server",
        "nosql", "firebase", "couchdb", "influxdb",
    ],
    "cloud_devops": [
        "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
        "k8s", "terraform", "ansible", "jenkins", "ci/cd", "circleci",
        "github actions", "gitlab ci", "helm", "prometheus", "grafana",
        "linux", "unix", "nginx", "apache", "vagrant", "puppet",
        "cloudformation", "lambda", "ec2", "s3", "ecs", "eks",
    ],
    "big_data": [
        "hadoop", "spark", "kafka", "hive", "pig", "flink", "airflow",
        "etl", "data pipeline", "data warehouse", "redshift", "snowflake",
        "databricks", "azure data factory", "glue", "nifi",
    ],
    "testing_qa": [
        "selenium", "junit", "pytest", "testng", "cypress", "jest",
        "mocha", "chai", "postman", "jmeter", "loadrunner", "appium",
        "cucumber", "bdd", "tdd", "unit testing", "integration testing",
        "regression testing", "automation testing", "manual testing",
        "api testing", "performance testing",
    ],
    "version_control_tools": [
        "git", "github", "gitlab", "bitbucket", "svn", "jira",
        "confluence", "trello", "asana", "slack", "notion",
        "visual studio code", "intellij", "eclipse", "pycharm",
    ],
    "security": [
        "cybersecurity", "network security", "penetration testing",
        "ethical hacking", "firewall", "vpn", "ssl", "tls", "oauth",
        "jwt", "encryption", "siem", "soc", "nmap", "wireshark",
        "metasploit", "burp suite", "kali linux", "iso 27001",
    ],
    "soft_skills": [
        "leadership", "communication", "teamwork", "problem solving",
        "analytical", "critical thinking", "project management",
        "agile", "scrum", "kanban", "time management", "presentation",
        "collaboration", "mentoring", "stakeholder management",
    ],
    "business_finance": [
        "excel", "powerpoint", "word", "sap", "erp", "crm", "salesforce",
        "financial modeling", "forecasting", "budgeting", "accounting",
        "quickbooks", "business analysis", "requirements gathering",
        "process improvement", "six sigma", "lean", "kpi",
    ],
    "mobile": [
        "android", "ios", "react native", "flutter", "xamarin",
        "objective-c", "mobile development", "app development",
    ],
    "blockchain": [
        "blockchain", "solidity", "ethereum", "smart contract",
        "web3", "nft", "defi", "hyperledger", "truffle", "metamask",
        "cryptocurrency", "bitcoin",
    ],
    "networking": [
        "tcp/ip", "dns", "dhcp", "routing", "switching", "cisco",
        "juniper", "lan", "wan", "sdwan", "bgp", "ospf", "mpls",
        "network administration", "ccna", "ccnp",
    ],
}

# Flattened list used internally for fast lookup
_ALL_SKILLS: list[str] = [
    skill
    for skills in SKILLS_REGISTRY.values()
    for skill in skills
]

# ATS-friendly section keywords — used when computing the ATS score
_ATS_SECTIONS = [
    "education", "experience", "skills", "summary", "objective",
    "certifications", "projects", "achievements", "publications",
    "awards", "languages", "interests", "references", "contact",
    "work experience", "professional experience", "technical skills",
    "academic", "profile",
]

# Resume quality signals used for suggestion generation
_QUALITY_SIGNALS = {
    "email":    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    "phone":    r"(\+?\d[\d\s\-().]{7,}\d)",
    "linkedin": r"linkedin\.com",
    "github":   r"github\.com",
    "url":      r"https?://[^\s]+",
    "metrics":  r"\b(\d+[\%\+x]|\d+\s*(years?|months?|projects?|clients?|teams?|members?))\b",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. PDF TEXT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(source) -> str:
    """
    Extract plain text from a PDF resume.

    Accepts multiple input types for flexibility across web frameworks:
      - str / os.PathLike : file path on disk
      - bytes             : raw PDF bytes (e.g. from an HTTP upload)
      - file-like object  : any object with a .read() method (BytesIO, FileStorage)

    Strategy:
      Iterates over every page and concatenates extracted text.
      Pages that yield no text (e.g. scanned images without OCR) are skipped
      gracefully; a warning is included in the output so the caller can
      surface it to the user.

    Parameters
    ----------
    source : str | PathLike | bytes | file-like
        The PDF to extract text from.

    Returns
    -------
    str
        Concatenated text from all extractable pages, or an empty string
        if extraction fails entirely.

    Raises
    ------
    RuntimeError
        If no PDF library is available in the environment.
    """
    if PyPDF2 is None:
        raise RuntimeError(
            "No PDF library found. Install pypdf: pip install pypdf"
        )

    # ── Normalise input to a file-like object ────────────────────────────────
    if isinstance(source, (str, os.PathLike)):
        with open(source, "rb") as fh:
            raw = fh.read()
        file_obj = io.BytesIO(raw)
    elif isinstance(source, bytes):
        file_obj = io.BytesIO(source)
    elif hasattr(source, "read"):
        # Flask FileStorage, Django InMemoryUploadedFile, or plain BytesIO
        content = source.read()
        file_obj = io.BytesIO(content)
    else:
        raise TypeError(
            f"Unsupported source type: {type(source)}. "
            "Pass a file path, bytes, or file-like object."
        )

    # ── Extract text page by page ─────────────────────────────────────────────
    extracted_pages: list[str] = []
    skipped_pages: list[int] = []

    try:
        reader = PyPDF2.PdfReader(file_obj)
        total_pages = len(reader.pages)

        for page_num, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
                page_text = page_text.strip()
                if page_text:
                    extracted_pages.append(page_text)
                else:
                    skipped_pages.append(page_num)
            except Exception:
                # A single unreadable page should not abort the whole document
                skipped_pages.append(page_num)
                continue

    except Exception as exc:
        # Return empty string rather than crashing — caller decides how to handle
        return ""

    full_text = "\n".join(extracted_pages)

    # Append a soft warning if some pages were image-only
    if skipped_pages and not full_text.strip():
        full_text = (
            "[WARNING: This PDF appears to be a scanned image. "
            "Text extraction returned no content. "
            "Please upload a text-based PDF for best results.]"
        )

    return full_text


# ─────────────────────────────────────────────────────────────────────────────
# 2. TEXT CLEANING (mirrors train_model.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

def fix_encoding(text: str) -> str:
    """
    Repair encoding artifacts from UTF-8 / Latin-1 mismatches.

    Attempts to re-interpret the string as Latin-1 bytes → UTF-8.
    Falls back to ASCII stripping if re-encoding fails.

    This function is kept identical to train_model.py to ensure the
    text seen during inference matches what the model was trained on.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """
    if not isinstance(text, str):
        return ""
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text.encode("ascii", errors="ignore").decode("ascii")


def clean_text(text: str) -> str:
    """
    Normalize resume text for TF-IDF vectorization.

    Operations (applied in order — identical to train_model.py):
      1. Fix encoding artifacts
      2. Lowercase
      3. Remove URLs
      4. Remove email addresses
      5. Remove non-alphanumeric characters (keep letters, digits, spaces)
      6. Collapse multiple whitespace → single space
      7. Strip leading / trailing whitespace

    Parameters
    ----------
    text : str
        Raw resume text.

    Returns
    -------
    str
        Cleaned, normalized text.
    """
    text = fix_encoding(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────────────────────────────────────
# 3. SKILL EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_skills(
    raw_text: str,
    skills_registry: Optional[dict[str, list[str]]] = None,
) -> dict[str, list[str]]:
    """
    Detect technical and soft skills mentioned in a resume.

    Performs case-insensitive, whole-word matching against the skills
    registry. Multi-word skills (e.g. "machine learning") are matched
    as phrases inside the original (lightly normalised) text to avoid
    false positives from individual word matches.

    Parameters
    ----------
    raw_text : str
        Original (un-cleaned) resume text, preserving punctuation context
        for phrase matching.
    skills_registry : dict[str, list[str]], optional
        Override the module-level SKILLS_REGISTRY with a custom dict.
        Each key is a domain name; values are lists of skill strings.

    Returns
    -------
    dict[str, list[str]]
        Mapping of domain → list of detected skills, e.g.:
        {
            "programming_languages": ["python", "java"],
            "data_science_ml": ["machine learning", "scikit-learn"],
            ...
        }
        Domains with zero matches are omitted.
    """
    registry = skills_registry or SKILLS_REGISTRY

    # Normalise for matching: lowercase, collapse whitespace, keep alphanumeric
    # and a few special chars that appear in skill names (e.g. "/" in "ci/cd")
    search_text = re.sub(r"\s+", " ", raw_text.lower())
    # Replace common separators with space so "c++" and "c #" don't confuse regex
    search_text_plain = re.sub(r"[^a-z0-9\s/#+.]", " ", search_text)

    detected: dict[str, list[str]] = {}

    for domain, skills in registry.items():
        found: list[str] = []
        for skill in skills:
            # Build a regex that matches the skill as a whole phrase
            # Use word boundaries for single-word skills; for multi-word
            # allow flexible whitespace between tokens
            pattern = r"\b" + r"\s+".join(re.escape(tok) for tok in skill.split()) + r"\b"
            if re.search(pattern, search_text_plain):
                found.append(skill)
        if found:
            detected[domain] = found

    return detected


def get_all_detected_skills(extracted: dict[str, list[str]]) -> list[str]:
    """
    Flatten the domain-keyed extraction result into a single sorted list.

    Parameters
    ----------
    extracted : dict[str, list[str]]
        Output of extract_skills().

    Returns
    -------
    list[str]
        Deduplicated, sorted list of all detected skills.
    """
    all_skills: set[str] = set()
    for skills in extracted.values():
        all_skills.update(skills)
    return sorted(all_skills)


# ─────────────────────────────────────────────────────────────────────────────
# 4. RESUME SCORE (out of 100)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_resume_score(
    raw_text: str,
    extracted_skills: dict[str, list[str]],
    target_category: Optional[str] = None,
) -> dict:
    """
    Score a resume out of 100 across four weighted dimensions.

    Scoring Rubric
    ──────────────
    Dimension              Weight   What it measures
    ─────────────────────  ──────   ──────────────────────────────────────────
    Skill breadth           40 pts  # distinct skills detected vs. benchmark
    Domain depth            25 pts  # domains covered (encourages versatility)
    Content richness        20 pts  Presence of contact info, links, metrics
    Resume length/detail    15 pts  Word count relative to ideal range

    Parameters
    ----------
    raw_text : str
        Original (un-cleaned) resume text.
    extracted_skills : dict[str, list[str]]
        Output of extract_skills().
    target_category : str, optional
        Job category hint (currently stored for future category-weighted scoring).

    Returns
    -------
    dict with keys:
        total_score     : int   (0–100)
        skill_score     : int   (0–40)
        domain_score    : int   (0–25)
        richness_score  : int   (0–20)
        length_score    : int   (0–15)
        total_skills    : int
        domains_covered : int
        breakdown       : dict  (per-dimension details)
    """
    all_skills = get_all_detected_skills(extracted_skills)
    total_skills = len(all_skills)
    domains_covered = len(extracted_skills)

    # ── Dimension 1: Skill breadth (0–40) ────────────────────────────────────
    # Benchmark: 30 skills = full marks (tuned empirically for typical resumes)
    SKILL_BENCHMARK = 30
    skill_score = min(40, round((total_skills / SKILL_BENCHMARK) * 40))

    # ── Dimension 2: Domain depth (0–25) ──────────────────────────────────────
    # Max possible domains in registry
    max_domains = len(SKILLS_REGISTRY)
    domain_score = min(25, round((domains_covered / max_domains) * 25))

    # ── Dimension 3: Content richness (0–20) ─────────────────────────────────
    richness_score = 0
    richness_breakdown: dict[str, bool] = {}

    signal_weights = {
        "email":    4,   # Contact info is essential
        "phone":    3,
        "linkedin": 4,   # Professional presence
        "github":   4,   # Portfolio (technical roles)
        "url":      2,   # Any other web presence
        "metrics":  3,   # Quantified achievements
    }
    for signal, weight in signal_weights.items():
        pattern = _QUALITY_SIGNALS[signal]
        found = bool(re.search(pattern, raw_text, re.IGNORECASE))
        richness_breakdown[signal] = found
        if found:
            richness_score += weight

    richness_score = min(20, richness_score)

    # ── Dimension 4: Resume length / detail (0–15) ───────────────────────────
    word_count = len(raw_text.split())
    # Ideal range: 300–900 words for a focused professional resume
    if word_count < 100:
        length_score = 2
    elif word_count < 200:
        length_score = 5
    elif word_count < 300:
        length_score = 8
    elif word_count <= 900:
        # Linear scale within ideal band
        length_score = 8 + round(((word_count - 300) / 600) * 7)
    elif word_count <= 1_500:
        length_score = 14   # Slightly long but acceptable
    else:
        length_score = 10   # Too verbose — penalise

    length_score = min(15, length_score)

    total_score = skill_score + domain_score + richness_score + length_score

    return {
        "total_score":      total_score,
        "skill_score":      skill_score,
        "domain_score":     domain_score,
        "richness_score":   richness_score,
        "length_score":     length_score,
        "total_skills":     total_skills,
        "domains_covered":  domains_covered,
        "word_count":       word_count,
        "breakdown": {
            "skill_benchmark":    SKILL_BENCHMARK,
            "max_domains":        max_domains,
            "richness_signals":   richness_breakdown,
            "target_category":    target_category,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. ATS COMPATIBILITY SCORE
# ─────────────────────────────────────────────────────────────────────────────

def calculate_ats_score(raw_text: str, extracted_skills: dict[str, list[str]]) -> dict:
    """
    Estimate how well a resume will fare against Applicant Tracking Systems.

    ATS systems parse resumes to extract structured data; they penalise:
      - Missing standard sections (Experience, Education, Skills, etc.)
      - Very short or very long documents
      - Absence of contact information
      - Lack of keywords / skills

    Scoring Rubric
    ──────────────
    Dimension              Weight
    ─────────────────────  ──────
    Standard sections       35 pts
    Keyword density         30 pts
    Contact completeness    20 pts
    Formatting signals      15 pts

    Parameters
    ----------
    raw_text : str
        Original resume text.
    extracted_skills : dict[str, list[str]]
        Output of extract_skills().

    Returns
    -------
    dict with keys:
        ats_score           : int (0–100)
        sections_score      : int (0–35)
        keyword_score       : int (0–30)
        contact_score       : int (0–20)
        formatting_score    : int (0–15)
        sections_found      : list[str]
        sections_missing    : list[str]
        contact_details     : dict[str, bool]
        ats_grade           : str  (A / B / C / D / F)
    """
    text_lower = raw_text.lower()

    # ── Sections score (0–35) ─────────────────────────────────────────────────
    # Weight core sections more heavily
    section_weights = {
        "experience":           7,
        "education":            7,
        "skills":               6,
        "summary":              4,
        "contact":              3,   # detected via pattern below
        "projects":             3,
        "certifications":       3,
        "achievements":         2,
    }

    sections_found: list[str] = []
    sections_missing: list[str] = []
    sections_score = 0

    # Check for email/phone as proxy for "contact" section
    has_contact = bool(
        re.search(_QUALITY_SIGNALS["email"], raw_text)
        or re.search(_QUALITY_SIGNALS["phone"], raw_text)
    )

    for section, weight in section_weights.items():
        if section == "contact":
            present = has_contact
        else:
            # Match section header as its own word/line
            present = bool(re.search(rf"\b{re.escape(section)}\b", text_lower))
            # Also accept plurals / common variants
            if not present and section == "experience":
                present = bool(re.search(r"\b(work experience|professional experience|employment)\b", text_lower))
            if not present and section == "certifications":
                present = bool(re.search(r"\b(certification|certificate|certified)\b", text_lower))
            if not present and section == "summary":
                present = bool(re.search(r"\b(objective|profile|overview|about)\b", text_lower))

        if present:
            sections_found.append(section)
            sections_score += weight
        else:
            sections_missing.append(section)

    sections_score = min(35, sections_score)

    # ── Keyword density score (0–30) ──────────────────────────────────────────
    total_detected = len(get_all_detected_skills(extracted_skills))
    # 20+ distinct skills → full marks
    keyword_score = min(30, round((total_detected / 20) * 30))

    # ── Contact completeness score (0–20) ─────────────────────────────────────
    contact_signals = {
        "email":    (bool(re.search(_QUALITY_SIGNALS["email"], raw_text)),    7),
        "phone":    (bool(re.search(_QUALITY_SIGNALS["phone"], raw_text)),    5),
        "linkedin": (bool(re.search(_QUALITY_SIGNALS["linkedin"], raw_text, re.I)), 5),
        "location": (bool(re.search(r"\b(city|state|country|remote|location|address|[a-z]+,\s*[a-z]{2}\b)", raw_text, re.I)), 3),
    }
    contact_score = sum(w for (found, w) in contact_signals.values() if found)
    contact_score = min(20, contact_score)
    contact_details = {k: v[0] for k, v in contact_signals.items()}

    # ── Formatting signals (0–15) ──────────────────────────────────────────────
    # ATS prefers plain text, consistent formatting, appropriate length
    formatting_score = 0
    word_count = len(raw_text.split())

    # Appropriate length (200–1000 words is ATS-safe)
    if 200 <= word_count <= 1_000:
        formatting_score += 6
    elif 100 <= word_count <= 1_500:
        formatting_score += 3

    # Presence of dates (indicates structured work history)
    if re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{4})\b", text_lower):
        formatting_score += 4

    # Presence of numbers / metrics (ATS parses these)
    if re.search(_QUALITY_SIGNALS["metrics"], raw_text, re.I):
        formatting_score += 3

    # No obvious table/image markers that confuse ATS
    # (PDFs that render cleanly won't have "----" artefacts)
    if not re.search(r"-{5,}|_{5,}|\|{3,}", raw_text):
        formatting_score += 2

    formatting_score = min(15, formatting_score)

    ats_score = sections_score + keyword_score + contact_score + formatting_score

    # Grade mapping
    if ats_score >= 85:
        grade = "A"
    elif ats_score >= 70:
        grade = "B"
    elif ats_score >= 55:
        grade = "C"
    elif ats_score >= 40:
        grade = "D"
    else:
        grade = "F"

    return {
        "ats_score":         ats_score,
        "sections_score":    sections_score,
        "keyword_score":     keyword_score,
        "contact_score":     contact_score,
        "formatting_score":  formatting_score,
        "sections_found":    sections_found,
        "sections_missing":  sections_missing,
        "contact_details":   contact_details,
        "ats_grade":         grade,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. IMPROVEMENT SUGGESTIONS
# ─────────────────────────────────────────────────────────────────────────────

def generate_suggestions(
    raw_text: str,
    score_result: dict,
    ats_result: dict,
    extracted_skills: dict[str, list[str]],
    predicted_category: Optional[str] = None,
) -> list[dict]:
    """
    Generate actionable, prioritised resume improvement suggestions.

    Each suggestion is a dict with:
      - priority : "high" | "medium" | "low"
      - category : short label (e.g. "Missing Section", "ATS", "Skills")
      - message  : concise, specific advice

    Suggestions are ordered by priority (high → medium → low) and then
    by estimated impact on the overall score.

    Parameters
    ----------
    raw_text : str
        Original resume text.
    score_result : dict
        Output of calculate_resume_score().
    ats_result : dict
        Output of calculate_ats_score().
    extracted_skills : dict[str, list[str]]
        Output of extract_skills().
    predicted_category : str, optional
        Predicted job category from the ML model.

    Returns
    -------
    list[dict]
        Prioritised list of suggestion dicts.
    """
    suggestions: list[dict] = []

    def add(priority: str, category: str, message: str) -> None:
        suggestions.append({"priority": priority, "category": category, "message": message})

    # ── Contact information ───────────────────────────────────────────────────
    contact = ats_result.get("contact_details", {})
    if not contact.get("email"):
        add("high", "Contact Info", "Add a professional email address. ATS systems require it for candidate identification.")
    if not contact.get("phone"):
        add("high", "Contact Info", "Include a phone number. Recruiters expect direct contact options.")
    if not contact.get("linkedin"):
        add("medium", "Professional Presence", "Add your LinkedIn profile URL. It significantly increases recruiter engagement.")
    if "github" not in raw_text.lower() and predicted_category in (
        "Data Science", "Python Developer", "Java Developer", "DevOps Engineer",
        "Web Designing", "Automation Testing", "Blockchain", "DotNet Developer",
        "Network Security Engineer", "Hadoop", "Database", "ETL Developer",
    ):
        add("medium", "Portfolio", f"Add a GitHub profile link. For '{predicted_category}' roles, a portfolio is highly valued.")

    # ── Missing resume sections ───────────────────────────────────────────────
    section_advice = {
        "summary":         ("medium", "Structure", "Add a professional summary (3–4 sentences) at the top. It's the first thing recruiters read."),
        "skills":          ("high",   "ATS",       "Add a dedicated 'Skills' section. ATS parsers specifically look for this heading."),
        "experience":      ("high",   "Structure", "Add a clearly labelled 'Work Experience' or 'Professional Experience' section."),
        "education":       ("high",   "Structure", "Add an 'Education' section. Most ATS systems require it for candidate filtering."),
        "projects":        ("low",    "Content",   "Add a 'Projects' section to showcase hands-on work, especially for technical roles."),
        "certifications":  ("low",    "Credibility", "Include any relevant certifications. They serve as third-party skill validation."),
        "achievements":    ("low",    "Impact",    "Add an 'Achievements' section with quantified results (%, $, time saved)."),
    }
    for section in ats_result.get("sections_missing", []):
        if section in section_advice:
            priority, category, message = section_advice[section]
            add(priority, category, message)

    # ── Skills gaps ───────────────────────────────────────────────────────────
    total_skills = score_result.get("total_skills", 0)
    if total_skills < 5:
        add("high", "Skills", "Only a few skills were detected. Expand your skills section with relevant technical and soft skills.")
    elif total_skills < 10:
        add("medium", "Skills", f"Only {total_skills} skills detected. Add more domain-specific tools and technologies to improve keyword matching.")
    elif total_skills < 15:
        add("low", "Skills", f"{total_skills} skills detected. Consider adding more niche or advanced skills relevant to your target role.")

    # Category-specific skill suggestions
    if predicted_category:
        category_skill_hints = {
            "Data Science":    ["tensorflow", "pytorch", "sql", "tableau", "statistics"],
            "Python Developer":["django", "flask", "fastapi", "pytest", "docker"],
            "Java Developer":  ["spring", "springboot", "maven", "junit", "microservices"],
            "DevOps Engineer": ["kubernetes", "terraform", "ansible", "jenkins", "prometheus"],
            "Web Designing":   ["react", "angular", "css3", "figma", "ux"],
            "HR":              ["recruitment", "talent acquisition", "hris", "performance management"],
            "Blockchain":      ["solidity", "ethereum", "smart contract", "web3", "hyperledger"],
            "Data Science":    ["pandas", "scikit-learn", "machine learning", "tableau", "sql"],
        }
        hints = category_skill_hints.get(predicted_category, [])
        all_found = get_all_detected_skills(extracted_skills)
        missing_hints = [h for h in hints if h not in all_found]
        if missing_hints:
            add("medium", "Role Alignment",
                f"For '{predicted_category}', consider adding: {', '.join(missing_hints[:4])}.")

    # ── Quantification & metrics ──────────────────────────────────────────────
    if not re.search(_QUALITY_SIGNALS["metrics"], raw_text, re.I):
        add("medium", "Impact", "Add quantified achievements (e.g. 'Reduced load time by 40%', 'Managed team of 8'). Numbers stand out to both ATS and recruiters.")

    # ── Resume length ─────────────────────────────────────────────────────────
    word_count = score_result.get("word_count", 0)
    if word_count < 200:
        add("high", "Content Depth", f"Resume is very short ({word_count} words). Expand work experience, projects, and skills to at least 300 words.")
    elif word_count < 300:
        add("medium", "Content Depth", f"Resume is brief ({word_count} words). Consider adding more detail to each role or project.")
    elif word_count > 1_500:
        add("low", "Conciseness", f"Resume is long ({word_count} words). Consider trimming to 600–900 words for better readability.")

    # ── Domain coverage ───────────────────────────────────────────────────────
    domains_covered = score_result.get("domains_covered", 0)
    if domains_covered < 2:
        add("medium", "Versatility", "Skills appear concentrated in one domain. Broadening to include soft skills, tools, or complementary technologies improves profile strength.")

    # ── ATS overall ──────────────────────────────────────────────────────────
    ats_score = ats_result.get("ats_score", 0)
    if ats_score < 40:
        add("high", "ATS", f"ATS score is low ({ats_score}/100). Focus on adding standard section headings, contact info, and more keywords.")
    elif ats_score < 60:
        add("medium", "ATS", f"ATS score could be improved ({ats_score}/100). Ensure section headings are clear and keywords match the job description.")

    # ── Sort by priority ──────────────────────────────────────────────────────
    priority_order = {"high": 0, "medium": 1, "low": 2}
    suggestions.sort(key=lambda s: priority_order[s["priority"]])

    return suggestions


# ─────────────────────────────────────────────────────────────────────────────
# 7. FULL ANALYSIS PIPELINE (convenience wrapper)
# ─────────────────────────────────────────────────────────────────────────────

def analyse_resume(
    raw_text: str,
    predicted_category: Optional[str] = None,
    skills_registry: Optional[dict[str, list[str]]] = None,
) -> dict:
    """
    Run the complete analysis pipeline on a resume text and return a
    single unified result dict.

    This is a convenience wrapper that calls, in order:
      1. extract_skills()
      2. calculate_resume_score()
      3. calculate_ats_score()
      4. generate_suggestions()

    Parameters
    ----------
    raw_text : str
        Original (un-cleaned) resume text extracted from PDF or plain text.
    predicted_category : str, optional
        Predicted job category from the ML classifier.
    skills_registry : dict, optional
        Custom skills registry to override the default.

    Returns
    -------
    dict
        {
            "clean_text"        : str,
            "extracted_skills"  : dict[str, list[str]],
            "all_skills"        : list[str],
            "resume_score"      : dict,
            "ats_result"        : dict,
            "suggestions"       : list[dict],
            "predicted_category": str | None,
        }
    """
    cleaned = clean_text(raw_text)
    extracted_skills = extract_skills(raw_text, skills_registry)
    all_skills = get_all_detected_skills(extracted_skills)
    resume_score = calculate_resume_score(raw_text, extracted_skills, predicted_category)
    ats_result = calculate_ats_score(raw_text, extracted_skills)
    suggestions = generate_suggestions(
        raw_text, resume_score, ats_result, extracted_skills, predicted_category
    )

    return {
        "clean_text":          cleaned,
        "extracted_skills":    extracted_skills,
        "all_skills":          all_skills,
        "resume_score":        resume_score,
        "ats_result":          ats_result,
        "suggestions":         suggestions,
        "predicted_category":  predicted_category,
    }
