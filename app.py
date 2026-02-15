import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
)

import matplotlib.pyplot as plt
# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="ML Model Comparator", layout="wide")

MODEL_DIR = Path("models")

MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logreg.pkl",
    "Decision Tree": MODEL_DIR / "dtree.pkl",
    "KNN (k=5)": MODEL_DIR / "knn.pkl",
    "Gaussian Naive Bayes": MODEL_DIR / "gnb.pkl",
    "Random Forest": MODEL_DIR / "rf.pkl",
    "XGBoost": MODEL_DIR / "xgb.pkl",
}

METRICS_CSV = Path("metrics.csv")
FEATURES_JSON = Path("feature_columns.json")

