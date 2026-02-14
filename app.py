import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, classification_report)

st.set_page_config(page_title="Dry Bean Identification", layout="wide")

sklearn_ver = sklearn.__version__
if not sklearn_ver.startswith("1.6"):
    st.warning(f"**Version Warning:** You are using sklearn {sklearn_ver}. "
               "Models were trained on 1.6.x. If you see errors, please "
               "re-run your training script.")

@st.cache_resource
def load_pkl_files(model_name):
    try:
        scaler = joblib.load('model/scaler.pkl')
        le = joblib.load('model/label_encoder.pkl')
        filename = f"model/{model_name.replace(' ', '_').lower()}.pkl"
        model = joblib.load(filename)
        return model, scaler, le
    except FileNotFoundError as ex:
        st.error(f"Error while loading the pickle files: {ex}")
        return None, None, None

st.title("Dry Bean Classification")
st.markdown("Upload the test data set for classification analysis")

st.sidebar.header("1. Data Input")
test_data_file = st.sidebar.file_uploader("Upload Test CSV", type=["csv"])

st.sidebar.header("2. Model Selection")
models = ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
model_name = st.sidebar.selectbox("Choose Model:", models)

if not test_data_file:
    st.info("Please upload a test CSV file in the sidebar to begin evaluation!")
    st.stop()

test_data = pd.read_csv(test_data_file)
model, scaler, le = load_pkl_files(model_name)

if 'Class' in test_data.columns:
    X = test_data.drop('Class', axis=1)
    y_raw = test_data['Class']

    X_scaled = scaler.transform(X)
    y_test = le.transform(y_raw)

    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)

    # Filter for present classes
    present_classes_index = np.unique(y_test)
    present_classes = [le.classes_[i] for i in present_classes_index]

    # Probabilities for present classes only (to avoid AUC warnings)
    y_proba_present = y_proba[:, present_classes_index]
    y_proba_present = y_proba_present / y_proba_present.sum(axis=1, keepdims=True)

    if len(present_classes_index) > 1:
        auc = roc_auc_score(y_test, y_proba_present, multi_class='ovr', 
                            average='weighted', labels=present_classes_index)
    else:
        auc = 0.0

    # Metrics
    precision = precision_score(y_test, y_pred, average='weighted', labels=present_classes_index, zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', labels=present_classes_index, zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', labels=present_classes_index, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    st.subheader(f"Performance Metrics for Model: {model_name}")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
    m2.metric("AUC", f"{auc:.4f}")
    m3.metric("Precision", f"{precision:.4f}")
    m4.metric("Recall", f"{recall:.4f}")
    m5.metric("F1 Score", f"{f1:.4f}")
    m6.metric("MCC Score", f"{mcc:.4f}")

    st.divider()
    cm_col, cp_col = st.columns(2)

    with cm_col:
        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred, labels=present_classes_index)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                    xticklabels=present_classes, yticklabels=present_classes)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        st.pyplot(fig)

    with cp_col:
        st.write("### Classification Report")
        report = classification_report(y_test, y_pred, labels=present_classes_index, 
                                       target_names=present_classes, output_dict=True, zero_division=0)
        st.table(pd.DataFrame(report).transpose().iloc[:-3, :])
else:
    st.error("Uploaded test CSV must have 'Class' column in it for evaluation")
