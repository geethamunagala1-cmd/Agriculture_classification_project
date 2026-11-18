import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# ---------------- Page configuration ----------------
st.set_page_config(page_title="Agriculture Risk Prediction", page_icon="🌱", layout="wide")
st.title("🌱 Agriculture Risk Category Prediction")
st.caption("Machine-learning application for predicting agricultural risk categories.")

TARGET = "risk_category"
DEFAULT_DATA = Path(__file__).parent / "modern_agriculture_ai_crop_production_2020_2050.csv"

# ---------------- Load data ----------------
uploaded_csv = st.sidebar.file_uploader("Upload the agriculture CSV dataset (optional)", type=["csv"])


@st.cache_data(show_spinner=False)
def load_csv_from_path(path):
    return pd.read_csv(
        path,
        nrows=15000,
        low_memory=True
    )


if uploaded_csv is not None:
    data = pd.read_csv(uploaded_csv)
elif DEFAULT_DATA.exists():
    data = load_csv_from_path(DEFAULT_DATA)
else:
    st.warning("Dataset file not found. Upload your CSV file using the sidebar.")
    st.stop()

if TARGET not in data.columns:
    st.error(f"Dataset must contain a '{TARGET}' column.")
    st.stop()


# ---------------- Prepare features and model ----------------
# Exclude target, ID, and recommended_action (which may reveal the risk label).
excluded_columns = [TARGET]
for col in ["record_id", "recommended_action"]:
    if col in data.columns:
        excluded_columns.append(col)

X = data.drop(columns=excluded_columns).copy()
y = data[TARGET].copy()

numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_transformer, numeric_features),
    ("categorical", categorical_transformer, categorical_features)
])

@st.cache_resource(show_spinner="Training model on dataset. This may take a moment...")
def train_model(data_for_training, feature_columns, target_values):
    # Cache based on the dataset contents passed by Streamlit.
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            class_weight="balanced", max_iter=1500, C=1.0,
            solver="lbfgs", random_state=42
        ))
    ])
    model.fit(data_for_training[feature_columns], target_values)
    return model


with st.spinner("Preparing model..."):
    model = train_model(X, X.columns.tolist(), y)


# ---------------- Dataset overview ----------------
with st.expander("Dataset overview", expanded=False):
    c1, c2, c3 = st.columns(3)
    c1.metric("Records", f"{len(data):,}")
    c2.metric("Input features", len(X.columns))
    c3.metric("Risk classes", y.nunique())
    st.write("Risk category distribution")
    st.bar_chart(y.value_counts())
    st.dataframe(data.head(), use_container_width=True)


# ---------------- Prediction form ----------------
st.subheader("Enter farm information")
st.info("Fill in the farm details below and click **Predict Risk Category**.")

with st.form("prediction_form"):
    user_input = {}
    # Organize inputs into columns for a compact form.
    left, right = st.columns(2)
    for i, feature in enumerate(X.columns):
        container = left if i % 2 == 0 else right
        with container:
            if feature in categorical_features:
                values = data[feature].dropna().astype(str).unique().tolist()
                values = sorted(values)
                if not values:
                    user_input[feature] = st.text_input(feature.replace("_", " ").title())
                else:
                    user_input[feature] = st.selectbox(
                        feature.replace("_", " ").title(), values
                    )
            else:
                series = pd.to_numeric(data[feature], errors="coerce")
                median = float(series.median()) if series.notna().any() else 0.0
                min_value = float(series.min()) if series.notna().any() else 0.0
                max_value = float(series.max()) if series.notna().any() else median + 100.0
                if min_value == max_value:
                    min_value = min_value - 1.0
                    max_value = max_value + 1.0
                user_input[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    min_value=min_value,
                    max_value=max_value,
                    value=min(max(median, min_value), max_value),
                    step=0.1,
                    format="%.3f"
                )
    predict_clicked = st.form_submit_button("Predict Risk Category", type="primary")


# ---------------- Prediction output ----------------
if predict_clicked:
    input_df = pd.DataFrame([user_input], columns=X.columns)
    prediction = model.predict(input_df)[0]
    st.divider()
    st.subheader("Prediction result")
    if str(prediction).lower() == "critical":
        st.error(f"Predicted Risk Category: **{prediction}**")
    elif str(prediction).lower() == "high":
        st.warning(f"Predicted Risk Category: **{prediction}**")
    else:
        st.success(f"Predicted Risk Category: **{prediction}**")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        probability_df = pd.DataFrame({"Risk category": model.classes_, "Probability": probabilities})
        probability_df = probability_df.sort_values("Probability", ascending=False)
        st.write("Prediction confidence by category")
        st.bar_chart(probability_df.set_index("Risk category"))
        st.dataframe(probability_df.assign(Probability=probability_df["Probability"].map(lambda p: f"{p:.1%}")), hide_index=True)

    if "recommended_action" in data.columns:
        matching = data.loc[data[TARGET] == prediction, "recommended_action"].dropna()
        if not matching.empty:
            recommendation = matching.mode().iloc[0]
            st.markdown("### Suggested action from dataset")
            st.write(recommendation)

st.caption("Note: This is a machine-learning prediction based on the dataset. It is not a substitute for expert agricultural assessment.")
