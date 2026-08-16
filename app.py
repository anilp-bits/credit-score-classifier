import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report



st.set_page_config(
    page_title="Credit Score Assessment Tool",
    page_icon="💳",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #F8FBFF;
}
</style>
""", unsafe_allow_html=True)


st.title("💳 Credit Score Classification")

st.write(
    "This application allows users to upload customer data and "
    "predict credit score categories using different trained models."
)


model_label = st.selectbox(
    "Select a model for prediction",
    (
        "Logistic Regression",
        "Decision Tree",
        "K-Nearest Neighbors",
        "Naive Bayes",
        "Random Forest"
    )
)


BASE_PATH = Path("model/saved_models")

MODEL_REGISTRY = {
    "Logistic Regression": BASE_PATH / "lr.pkl",
    "Decision Tree": BASE_PATH / "dt.pkl",
    "K-Nearest Neighbors": BASE_PATH / "knn.pkl",
    "Naive Bayes": BASE_PATH / "nb.pkl",
    "Random Forest": BASE_PATH / "rf.pkl"
}


@st.cache_resource
def load_pipeline():
    models = {}

    for name, path in MODEL_REGISTRY.items():
        models[name] = joblib.load(path)

    scaler = joblib.load(BASE_PATH / "scaler.pkl")
    encoders = joblib.load(BASE_PATH / "encoders.pkl")
    features = joblib.load(BASE_PATH / "features.pkl")

    return models, scaler, encoders, features


model_store, scaler_tool, encoder_store, feature_order = load_pipeline()
active_model = model_store[model_label]


def transform_dataset(input_df):
    working_df = input_df.copy()

    for column_name, encoder in encoder_store.items():
        if column_name in working_df.columns:
            working_df[column_name] = working_df[column_name].astype(str)

            working_df[column_name] = working_df[column_name].apply(
                lambda item: encoder.transform([item])[0]
                if item in encoder.classes_ else 0
            )

    working_df = working_df.reindex(columns=feature_order, fill_value=0)

    scaled_data = scaler_tool.transform(working_df)

    return scaled_data



st.subheader("Upload Dataset")

uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])



if "is_running" not in st.session_state:
    st.session_state.is_running = False



if uploaded_file is not None:

    raw_data = pd.read_csv(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Data Preview")
        st.dataframe(raw_data.head(), use_container_width=True)

    with col2:
        st.subheader("Dataset Summary")
        st.write(f"Rows: {raw_data.shape[0]}")
        st.write(f"Columns: {raw_data.shape[1]}")

    run_clicked = st.button(
        "Start Prediction",
        disabled=st.session_state.is_running
    )

    if run_clicked:
        st.session_state.is_running = True

        with st.spinner("⏳ Running model... Please wait"):

            try:
                model_input = transform_dataset(raw_data)

                predictions = active_model.predict(model_input)

                result_df = raw_data.copy()
                result_df["Predicted_Credit_Score"] = predictions

                st.success(f"Prediction completed using {model_label}")


                st.subheader("Prediction Results")
                st.dataframe(result_df, use_container_width=True)


                st.subheader("Prediction Distribution")

                c1, c2, c3 = st.columns([1, 2, 1])

                with c2:
                    st.bar_chart(
                        result_df["Predicted_Credit_Score"].value_counts()
                    )


                if "Credit_Score" in result_df.columns:

                    st.subheader("Confusion Matrix")

                    y_true = result_df["Credit_Score"]
                    y_pred = predictions

                    matrix = confusion_matrix(y_true, y_pred)

                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(
                        matrix,
                        annot=True,
                        fmt="d",
                        cmap="coolwarm",
                        ax=ax
                    )

                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    ax.set_title(f"{model_label} Performance")

                    c1, c2, c3 = st.columns([1, 2, 1])

                    with c2:
                        st.pyplot(fig)

                    st.subheader("Classification Report")
                    st.text(classification_report(y_true, y_pred))

                else:
                    st.warning(
                        "Actual labels not found → Confusion Matrix skipped"
                    )


                csv_data = result_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download Results",
                    csv_data,
                    "credit_predictions.csv",
                    "text/csv"
                )

            except Exception as error:
                st.error(f"Error occurred: {error}")

        st.session_state.is_running = False

else:
    st.info("Upload a CSV file to begin.")



st.markdown("---")