import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, classification_report)


@st.cache_resource
def load_pkl_files(model_name):
    try:
        scaler = joblib.load('model/scaler.pkl')
        le = joblib.load('model/label_encoder.pkl')
        model = joblib.load(f'model/{model_name.replace(' ', '_').lower()}.pkl')
        return model, scaler, le
    except FileNotFoundError as ex:
        st.error(f"Error while loading the pickle files: {ex}")
        return None, None, None


st.set_page_config(page_title="Dry Bean Identification based on Colour and Size", layout="wide")

st.title("Dry Bean Classification")
st.markdown("Upload the test data set for classification analysis")

st.sidebar.header("1. Data Input")
test_data_file = st.sidebar.file_uploader("Upload Test CSV", type=["csv"])

st.sidebar.header("2. Model Selection")
models = ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
model_name = st.sidebar.selectbox("Choose Model:", models)

if not test_data_file:
    st.info("Please upload test CSV file in the sidebar to begin evaluation!")

test_data = pd.read_csv(test_data_file)
model, scaler, le = load_pkl_files(model_name)

if 'Class' in test_data.columns:
    X = test_df.drop('Class', axis=1)
    y_raw = test_df['Class']

    X_scaled = scaler.transform(X)
    y_test = le.transform(y_test_raw)

    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_test_scaled)

    # Get present clases in the test data
    present_classes_index = np.unique(y_test)
    present_classes = [le.classes_[i] for i in present_classes_index]

    # Get probabilites for present classes only
    y_proba_present_classes = y_proba[:, present_classes_index]
    y_proba_present_classes = y_proba_present_classes / y_proba_present_classes.sum(axis=1, keepdims=True)

    if len(present_classes_index) > 1:
        auc = roc_auc_score(y_test, y_proba_present_classes, multi_class='ovr', average='weighted', labels=present_classes_index)
    else:
        auc = 0.0

    st.subheader(f"Performance Metrics for Model: {model_name}")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4%}")
    m2.metric("AUC", f"{auc:.4f}")
    m3.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted'):.4f}")
    m4.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted'):.4f}")
    m5.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted'):.4f}")
    m6.metric("MCC Score", f"{matthews_corrcoef(y_test, y_pred):.4f}")

    st.divider()
    cm_col, cp_col = st.coulmns([1, 1])

    with cm_col:
        st.write("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred, labels=present_classes_index)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                    xticklabels=present_classes, yticklabels=present_classes)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        st.pyplot(fig)

    with cp_col:
        st.write("Classification Report")
        report = classification_report(y_test, y_pred, labels=present_classes_index, target_names=present_classes, output_dict=True)
        st.table(pd.DataFrame(report).transpose().iloc[:-3, :3])
else:
    st.error("Uploaded test CSV must have 'Class' column in it for evaluation")
