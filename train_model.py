"""
train_model.py
==============
AI Resume Screening System — Model Training Pipeline

This script handles the full ML pipeline:
  1. Data loading
  2. Preprocessing & cleaning (deduplication, encoding fixes, text normalization)
  3. Feature extraction via TF-IDF
  4. Model training using Multinomial Naive Bayes
  5. Evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)
  6. Persisting the trained model and vectorizer via Joblib

Author : Resume Screening System
Dataset: Resume_Screening.csv (962 rows, 25 job categories)
"""

import re
import string
import warnings
import os

import pandas as pd
import joblib
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

DATA_PATH = "dataset/Resume_Screening.csv"  # Path to the raw dataset
MODEL_PATH      = "resume_model.joblib"    # Output path for the trained model
VECTORIZER_PATH = "tfidf_vectorizer.joblib"  # Output path for the TF-IDF vectorizer
ENCODER_PATH    = "label_encoder.joblib"   # Output path for the label encoder

TEST_SIZE       = 0.20   # 80% train / 20% test
RANDOM_STATE    = 42     # Reproducibility seed

# TF-IDF hyperparameters
TFIDF_MAX_FEATURES = 15_000   # Vocabulary cap — keeps only top N terms by frequency
TFIDF_NGRAM_RANGE  = (1, 2)   # Unigrams + bigrams for richer feature representation
TFIDF_MIN_DF       = 2        # Ignore terms appearing in fewer than 2 documents
TFIDF_SUBLINEAR_TF = True     # Apply log(1 + tf) to dampen high-frequency terms


# ─────────────────────────────────────────────
# STEP 1: DATA LOADING
# ─────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """
    Load the raw CSV dataset and perform a basic integrity check.

    Parameters
    ----------
    path : str
        File path to the CSV dataset.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with 'Category' and 'Resume' columns.
    """
    print(f"\n{'='*60}")
    print("  STEP 1: Loading Data")
    print(f"{'='*60}")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Please place Resume_Screening.csv in the same directory as this script."
        )

    df = pd.read_csv(path)

    print(f"  ✔ Loaded   : {len(df):,} rows × {len(df.columns)} columns")
    print(f"  ✔ Columns  : {list(df.columns)}")
    print(f"  ✔ Categories: {df['Category'].nunique()} unique job roles")

    return df


# ─────────────────────────────────────────────
# STEP 2: PREPROCESSING
# ─────────────────────────────────────────────

def fix_encoding(text: str) -> str:
    """
    Repair common encoding artifacts that arise from UTF-8 / Latin-1 mismatches.

    Common patterns in the dataset:
      Ã¢  → â    (from â being misread as two Latin-1 chars)
      Ã©  → é
      Ã   → Â
      â€™ → '
      â€" → –
      Ã¯  → ï

    The approach: attempt to re-encode the mangled string back to bytes
    and decode with the correct codec. Fall back to aggressive regex
    removal if re-encoding fails.

    Parameters
    ----------
    text : str
        Raw resume text, potentially containing encoding artifacts.

    Returns
    -------
    str
        Text with encoding artifacts corrected where possible.
    """
    if not isinstance(text, str):
        return ""

    # Attempt 1: Re-interpret as Latin-1 bytes → UTF-8
    try:
        fixed = text.encode("latin-1").decode("utf-8")
        return fixed
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass

    # Attempt 2: Strip non-ASCII entirely as a safe fallback
    # This removes artifacts like Ã, â, ¢ that couldn't be decoded
    cleaned = text.encode("ascii", errors="ignore").decode("ascii")
    return cleaned


def clean_text(text: str) -> str:
    """
    Normalize resume text for TF-IDF feature extraction.

    Operations applied in order:
      1. Fix encoding artifacts
      2. Lowercase the entire text
      3. Remove URLs (http/https/www links)
      4. Remove email addresses
      5. Remove special characters, retaining only letters, digits, and spaces
      6. Collapse multiple whitespace into a single space
      7. Strip leading / trailing whitespace

    Parameters
    ----------
    text : str
        Raw or partially cleaned resume text.

    Returns
    -------
    str
        Normalized, clean text ready for vectorization.
    """
    # Fix encoding artifacts first
    text = fix_encoding(text)

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove non-alphanumeric characters (keep letters, digits, and spaces)
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
      - Drop null values in either column
      - Remove duplicate resume texts
      - Clean and normalize resume text
      - Filter out resumes that are too short after cleaning

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe with 'Category' and 'Resume' columns.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with an additional 'clean_resume' column.
    """
    print(f"\n{'='*60}")
    print("  STEP 2: Preprocessing & Cleaning")
    print(f"{'='*60}")

    initial_count = len(df)

    # 2a. Drop rows with missing values
    df = df.dropna(subset=["Category", "Resume"]).copy()
    after_null_drop = len(df)
    print(f"  ✔ Null rows removed      : {initial_count - after_null_drop}")

    # 2b. Remove exact duplicate resume texts
    # Duplicates are a significant issue in this dataset (~83% of entries)
    df = df.drop_duplicates(subset=["Resume"]).copy()
    after_dedup = len(df)
    print(f"  ✔ Duplicate rows removed : {after_null_drop - after_dedup}")

    # 2c. Clean and normalize resume text
    print("  ⏳ Cleaning resume text (encoding fix, normalization)...")
    df["clean_resume"] = df["Resume"].apply(clean_text)

    # 2d. Filter out resumes that are still too short after cleaning
    # A meaningful resume should have at least 50 characters after normalization
    min_length = 50
    before_short = len(df)
    df = df[df["clean_resume"].str.len() >= min_length].copy()
    print(f"  ✔ Too-short resumes removed: {before_short - len(df)} (< {min_length} chars after cleaning)")

    print(f"\n  📊 Dataset shape after cleaning: {len(df):,} rows")
    print(f"  📊 Categories retained: {df['Category'].nunique()}")
    print(f"\n  Category distribution (post-cleaning):")
    dist = df["Category"].value_counts()
    for cat, count in dist.items():
        bar = "█" * (count // 1)
        print(f"    {cat:<30} {count:>3}  {bar}")

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# STEP 3: FEATURE EXTRACTION (TF-IDF)
# ─────────────────────────────────────────────

def build_features(df: pd.DataFrame):
    """
    Transform cleaned resume text into numerical TF-IDF feature vectors
    and encode the target labels as integers.

    TF-IDF (Term Frequency–Inverse Document Frequency) assigns higher
    weight to terms that are frequent in a document but rare across the
    corpus — ideal for distinguishing domain-specific resume vocabulary
    (e.g. 'kubernetes' for DevOps vs 'tort' for Advocate).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe with 'clean_resume' and 'Category' columns.

    Returns
    -------
    X : scipy sparse matrix
        TF-IDF feature matrix.
    y : np.ndarray
        Integer-encoded category labels.
    vectorizer : TfidfVectorizer
        Fitted vectorizer (needed for inference).
    encoder : LabelEncoder
        Fitted label encoder (needed to decode predictions back to category names).
    """
    print(f"\n{'='*60}")
    print("  STEP 3: Feature Extraction — TF-IDF")
    print(f"{'='*60}")

    # Encode category labels as integers (required by MultinomialNB)
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["Category"])
    print(f"  ✔ Label encoder fitted: {len(encoder.classes_)} classes")

    # Build TF-IDF feature matrix
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,     # Include bigrams for phrases like 'machine learning'
        min_df=TFIDF_MIN_DF,               # Ignore very rare terms (likely noise or typos)
        sublinear_tf=TFIDF_SUBLINEAR_TF,   # Log-scale TF to reduce dominance of common terms
        stop_words="english",              # Remove English stop words via sklearn's built-in list
    )

    X = vectorizer.fit_transform(df["clean_resume"])

    print(f"  ✔ Vocabulary size       : {len(vectorizer.vocabulary_):,} terms")
    print(f"  ✔ Feature matrix shape  : {X.shape[0]:,} samples × {X.shape[1]:,} features")
    print(f"  ✔ Matrix sparsity       : {(1 - X.nnz / (X.shape[0] * X.shape[1])):.2%}")

    return X, y, vectorizer, encoder


# ─────────────────────────────────────────────
# STEP 4: TRAIN / TEST SPLIT
# ─────────────────────────────────────────────

def split_data(X, y):
    """
    Split features and labels into stratified train and test sets.

    Stratification ensures each job category is proportionally
    represented in both the train and test splits — important for
    class-imbalanced datasets.

    Parameters
    ----------
    X : scipy sparse matrix
        TF-IDF feature matrix.
    y : np.ndarray
        Integer-encoded category labels.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    print(f"\n{'='*60}")
    print("  STEP 4: Train / Test Split")
    print(f"{'='*60}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,    # Preserve class distribution across both splits
    )

    print(f"  ✔ Training samples : {X_train.shape[0]:,}  ({1 - TEST_SIZE:.0%})")
    print(f"  ✔ Test samples     : {X_test.shape[0]:,}  ({TEST_SIZE:.0%})")

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# STEP 5: MODEL TRAINING
# ─────────────────────────────────────────────

def train_model(X_train, y_train) -> MultinomialNB:
    """
    Train a Multinomial Naive Bayes classifier on the TF-IDF features.

    Why Multinomial Naive Bayes?
    - Designed for text classification with discrete (count / frequency) features
    - Efficient and fast even on high-dimensional sparse matrices
    - Strong baseline for short-to-medium text documents
    - Works well with TF-IDF when combined with sublinear_tf scaling

    alpha (Laplace smoothing) = 0.1 slightly reduces the default smoothing,
    which tends to improve precision on well-separated categories like
    technical domains (Python Developer vs. Advocate have very distinct vocab).

    Parameters
    ----------
    X_train : scipy sparse matrix
        Training feature matrix.
    y_train : np.ndarray
        Training labels.

    Returns
    -------
    MultinomialNB
        Fitted classifier.
    """
    print(f"\n{'='*60}")
    print("  STEP 5: Model Training — Multinomial Naive Bayes")
    print(f"{'='*60}")

    model = MultinomialNB(alpha=0.1)
    model.fit(X_train, y_train)

    print(f"  ✔ Model trained on {X_train.shape[0]:,} samples")
    print(f"  ✔ alpha (Laplace smoothing) : 0.1")

    return model


# ─────────────────────────────────────────────
# STEP 6: EVALUATION
# ─────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, encoder: LabelEncoder) -> None:
    """
    Evaluate the trained model on the held-out test set and print
    a comprehensive performance report.

    Metrics reported:
      - Accuracy  : Overall fraction of correct predictions
      - Precision : Of all predicted positives, how many are truly positive
      - Recall    : Of all actual positives, how many were correctly predicted
      - F1 Score  : Harmonic mean of Precision and Recall (macro-averaged)
      - Confusion Matrix : Grid of true vs predicted labels
      - Classification Report : Per-class breakdown of all metrics

    Parameters
    ----------
    model : MultinomialNB
        Trained classifier.
    X_test : scipy sparse matrix
        Test feature matrix.
    y_test : np.ndarray
        True labels for the test set.
    encoder : LabelEncoder
        Fitted encoder to decode integer labels back to category names.
    """
    print(f"\n{'='*60}")
    print("  STEP 6: Model Evaluation")
    print(f"{'='*60}")

    y_pred = model.predict(X_test)

    # ── Core metrics ──
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall    = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1        = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"\n  ┌─────────────────────────────────────┐")
    print(f"  │         PERFORMANCE METRICS          │")
    print(f"  ├─────────────────────────────────────┤")
    print(f"  │  Accuracy           :  {accuracy:.4f}        │")
    print(f"  │  Precision (macro)  :  {precision:.4f}        │")
    print(f"  │  Recall    (macro)  :  {recall:.4f}        │")
    print(f"  │  F1 Score  (macro)  :  {f1:.4f}        │")
    print(f"  └─────────────────────────────────────┘")

    # ── Classification Report (per-category breakdown) ──
    target_names = encoder.classes_
    print(f"\n  {'─'*60}")
    print("  CLASSIFICATION REPORT (per category)")
    print(f"  {'─'*60}")
    print(classification_report(
        y_test, y_pred,
        target_names=target_names,
        zero_division=0
    ))

    # ── Confusion Matrix ──
    cm = confusion_matrix(y_test, y_pred)
    print(f"  {'─'*60}")
    print("  CONFUSION MATRIX")
    print(f"  {'─'*60}")
    print(f"  Shape: {cm.shape[0]} × {cm.shape[1]} (categories × categories)")

    # Find misclassification pairs for insight
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)
    if cm_copy.max() > 0:
        top_errors = []
        for _ in range(min(5, cm_copy.sum() > 0 and 5 or 0)):
            idx = np.unravel_index(cm_copy.argmax(), cm_copy.shape)
            if cm_copy[idx] == 0:
                break
            true_cat  = target_names[idx[0]]
            pred_cat  = target_names[idx[1]]
            top_errors.append((true_cat, pred_cat, cm_copy[idx]))
            cm_copy[idx] = 0

        if top_errors:
            print(f"\n  Top misclassifications (true → predicted : count):")
            for true_cat, pred_cat, count in top_errors:
                print(f"    {true_cat:<30} → {pred_cat:<30} : {count}")

    print(f"\n  Diagonal (correct predictions per category):")
    for i, cat in enumerate(target_names):
        total = cm[i].sum()
        correct = cm[i, i]
        pct = correct / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 5)
        print(f"    {cat:<30} {correct:>2}/{total:<3} ({pct:5.1f}%)  {bar}")


# ─────────────────────────────────────────────
# STEP 7: SAVE ARTIFACTS
# ─────────────────────────────────────────────

def save_artifacts(model, vectorizer, encoder) -> None:
    """
    Persist the trained model, TF-IDF vectorizer, and label encoder to disk
    using Joblib (efficient serialization for numpy-backed objects).

    Three files are saved:
      - resume_model.joblib      : The fitted MultinomialNB classifier
      - tfidf_vectorizer.joblib  : The fitted TfidfVectorizer
      - label_encoder.joblib     : The fitted LabelEncoder

    All three are required at inference time to:
      1. Vectorize a new resume with the same vocabulary (vectorizer)
      2. Predict the job category integer label (model)
      3. Map the integer back to a human-readable category (encoder)

    Parameters
    ----------
    model       : MultinomialNB     — Trained classifier
    vectorizer  : TfidfVectorizer   — Fitted vectorizer
    encoder     : LabelEncoder      — Fitted label encoder
    """
    print(f"\n{'='*60}")
    print("  STEP 7: Saving Artifacts")
    print(f"{'='*60}")

    joblib.dump(model,      MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(encoder,    ENCODER_PATH)

    for path in [MODEL_PATH, VECTORIZER_PATH, ENCODER_PATH]:
        size_kb = os.path.getsize(path) / 1024
        print(f"  ✔ Saved: {path:<35} ({size_kb:.1f} KB)")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    """
    Orchestrates the full training pipeline end-to-end:
      load → preprocess → features → split → train → evaluate → save
    """
    print("\n" + "═" * 60)
    print("   AI RESUME SCREENING SYSTEM — MODEL TRAINING")
    print("═" * 60)

    # Step 1: Load raw data
    df = load_data(DATA_PATH)

    # Step 2: Clean and preprocess
    df = preprocess(df)

    # Step 3: Extract TF-IDF features and encode labels
    X, y, vectorizer, encoder = build_features(df)

    # Step 4: Split into train / test sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 5: Train the Multinomial Naive Bayes model
    model = train_model(X_train, y_train)

    # Step 6: Evaluate on the held-out test set
    evaluate_model(model, X_test, y_test, encoder)

    # Step 7: Save model, vectorizer, and encoder to disk
    save_artifacts(model, vectorizer, encoder)

    print(f"\n{'═' * 60}")
    print("   ✅ Training complete. Artifacts saved.")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
