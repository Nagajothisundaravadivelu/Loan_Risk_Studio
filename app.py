from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


TARGET = "not.fully.paid"
CATEGORICAL_FEATURES = ["purpose"]
NUMERIC_FEATURES = [
    "credit.policy",
    "int.rate",
    "installment",
    "log.annual.inc",
    "dti",
    "fico",
    "days.with.cr.line",
    "revol.bal",
    "revol.util",
    "inq.last.6mths",
    "delinq.2yrs",
    "pub.rec",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
INTEGER_FEATURES = {
    "credit.policy",
    "fico",
    "revol.bal",
    "inq.last.6mths",
    "delinq.2yrs",
    "pub.rec",
}
FIELD_LABELS = {
    "int.rate": "Interest rate",
    "installment": "Monthly installment",
    "log.annual.inc": "Log annual income",
    "dti": "Debt-to-income ratio",
    "fico": "FICO score",
    "days.with.cr.line": "Days with credit line",
    "revol.bal": "Revolving balance",
    "revol.util": "Revolving utilization (%)",
    "inq.last.6mths": "Credit inquiries (last 6 months)",
    "delinq.2yrs": "Delinquencies (last 2 years)",
    "pub.rec": "Public records",
}
FIELD_HELP = {
    "int.rate": "Enter a proportion: 0.12 means a 12% interest rate.",
    "log.annual.inc": "Annual income is represented as its natural logarithm, matching the training data.",
    "dti": "Use the dataset scale, where 12 means 12%.",
    "revol.util": "Use the dataset scale, where 50 means 50% utilization.",
}
MODEL_LABELS = {
    "Decision tree": "Decision tree",
    "Random forest": "Random forest (200 trees)",
}

st.set_page_config(page_title="Loan Risk Studio", page_icon="L", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: #f4f6f2; }
    [data-testid="stSidebar"] { background: #e9eee8; }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #d8dfd7;
        border-left: 3px solid #bd5639;
        padding: 12px 14px;
        border-radius: 4px;
    }
    .loan-note {
        border-left: 4px solid #bd5639;
        padding: 10px 14px;
        background: #fff;
        color: #34413a;
        margin: 6px 0 18px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_training_data(csv_path: str, modified_ns: int) -> pd.DataFrame:
    return pd.read_csv(csv_path)


@st.cache_resource(show_spinner="Training the models on the loan data...")
def train_models(training_data: pd.DataFrame) -> tuple[dict[str, Pipeline], dict[str, dict[str, float]]]:
    missing = [column for column in FEATURES + [TARGET] if column not in training_data.columns]
    if missing:
        raise ValueError(f"Training CSV is missing required columns: {', '.join(missing)}")

    clean_data = training_data[FEATURES + [TARGET]].dropna()
    X = clean_data[FEATURES]
    y = clean_data[TARGET].astype(int)
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=101, stratify=stratify
    )

    classifiers = {
        "Decision tree": DecisionTreeClassifier(random_state=101),
        "Random forest": RandomForestClassifier(
            n_estimators=200, random_state=101, n_jobs=-1
        ),
    }
    models: dict[str, Pipeline] = {}
    metrics: dict[str, dict[str, float]] = {}

    for name, classifier in classifiers.items():
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "purpose",
                    OneHotEncoder(drop="first", handle_unknown="ignore"),
                    CATEGORICAL_FEATURES,
                )
            ],
            remainder="passthrough",
        )
        model = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", classifier)]
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        models[name] = model
        metrics[name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
            "risk_recall": float(recall_score(y_test, predictions, zero_division=0)),
        }

    return models, metrics


def make_numeric_input(data: pd.DataFrame, column: str) -> int | float:
    values = pd.to_numeric(data[column], errors="coerce").dropna()
    if values.empty:
        raise ValueError(f"No numeric values found for {column}.")

    minimum = float(values.min())
    maximum = float(values.max())
    default = float(values.median())
    if column in INTEGER_FEATURES:
        return st.number_input(
            FIELD_LABELS[column],
            min_value=int(minimum),
            max_value=int(maximum),
            value=int(round(default)),
            step=1,
            key=f"input_{column}",
        )

    step = 1.0 if column in {"installment", "days.with.cr.line"} else 0.01
    return st.number_input(
        FIELD_LABELS[column],
        min_value=minimum,
        max_value=maximum,
        value=default,
        step=step,
        format="%.4f",
        help=FIELD_HELP.get(column),
        key=f"input_{column}",
    )


def risk_probability(model: Pipeline, features: pd.DataFrame) -> float:
    probabilities = model.predict_proba(features)[0]
    positive_class_index = list(model.classes_).index(1)
    return float(probabilities[positive_class_index])


st.title("Loan Risk Studio")
st.caption("LendingClub historical loan data · Decision tree and random forest")
st.markdown(
    '<div class="loan-note"><strong>Coursework use only.</strong> This historical, imbalanced dataset is not validated for real lending or credit decisions.</div>',
    unsafe_allow_html=True,
)

data_path = Path(__file__).resolve().with_name("loan_data.csv")
if not data_path.exists():
    st.error(f"Could not find {data_path.name}. Place it in the same folder as app.py.")
    st.stop()

try:
    training_data = load_training_data(str(data_path), data_path.stat().st_mtime_ns)
    models, model_metrics = train_models(training_data)
except Exception as error:
    st.error(f"Unable to load or train from loan_data.csv: {error}")
    st.stop()

with st.sidebar:
    st.header("Model settings")
    selected_model_name = st.radio(
        "Classifier",
        options=list(MODEL_LABELS),
        format_func=lambda name: MODEL_LABELS[name],
    )
    selected_model = models[selected_model_name]
    selected_metrics = model_metrics[selected_model_name]
    st.divider()
    st.metric("Holdout balanced accuracy", f"{selected_metrics['balanced_accuracy']:.1%}")
    st.metric("Holdout accuracy", f"{selected_metrics['accuracy']:.1%}")
    st.metric("Recall: not fully paid", f"{selected_metrics['risk_recall']:.1%}")
    st.caption("Metrics use a 30% holdout split. Recall shows the share of not-fully-paid loans the model found.")

fully_paid_share = 1.0 - float(training_data[TARGET].mean())
summary_columns = st.columns(3)
summary_columns[0].metric("Historical loans", f"{len(training_data):,}")
summary_columns[1].metric("Features", str(len(FEATURES)))
summary_columns[2].metric("Paid in full in sample", f"{fully_paid_share:.1%}")

single_tab, batch_tab = st.tabs(["Score one loan", "Score a CSV"])

with single_tab:
    st.subheader("Loan details")
    st.caption("Inputs are limited to the ranges present in the training data.")
    with st.form("single_loan_form"):
        purpose_values = sorted(training_data["purpose"].dropna().astype(str).unique())
        purpose_default = "debt_consolidation" if "debt_consolidation" in purpose_values else purpose_values[0]
        form_columns = st.columns(3)
        loan_features: dict[str, int | float | str] = {}

        with form_columns[0]:
            loan_features["credit.policy"] = st.selectbox(
                "Meets credit policy?",
                options=[1, 0],
                format_func=lambda value: "Yes" if value == 1 else "No",
            )
            loan_features["int.rate"] = make_numeric_input(training_data, "int.rate")
            loan_features["installment"] = make_numeric_input(training_data, "installment")
            loan_features["log.annual.inc"] = make_numeric_input(training_data, "log.annual.inc")

        with form_columns[1]:
            loan_features["purpose"] = st.selectbox(
                "Loan purpose",
                options=purpose_values,
                index=purpose_values.index(purpose_default),
            )
            loan_features["dti"] = make_numeric_input(training_data, "dti")
            loan_features["fico"] = make_numeric_input(training_data, "fico")
            loan_features["days.with.cr.line"] = make_numeric_input(training_data, "days.with.cr.line")

        with form_columns[2]:
            loan_features["revol.bal"] = make_numeric_input(training_data, "revol.bal")
            loan_features["revol.util"] = make_numeric_input(training_data, "revol.util")
            loan_features["inq.last.6mths"] = make_numeric_input(training_data, "inq.last.6mths")
            loan_features["delinq.2yrs"] = make_numeric_input(training_data, "delinq.2yrs")
            loan_features["pub.rec"] = make_numeric_input(training_data, "pub.rec")

        submitted = st.form_submit_button("Predict repayment class", type="primary", use_container_width=True)

    if submitted:
        record = pd.DataFrame([loan_features], columns=FEATURES)
        predicted_class = int(selected_model.predict(record)[0])
        probability = risk_probability(selected_model, record)
        result_columns = st.columns([1, 2])
        result_columns[0].metric(
            "Predicted class",
            "Not fully paid" if predicted_class == 1 else "Fully paid",
        )
        result_columns[1].metric("Model-estimated probability of not fully paid", f"{probability:.1%}")
        st.progress(probability)
        st.caption("The model's default class boundary is 50%. The probability is an uncalibrated model estimate, not a guarantee.")

with batch_tab:
    st.subheader("Batch predictions")
    st.caption("Upload a CSV with the same feature columns as loan_data.csv. The target column is optional and ignored for prediction.")
    with st.form("batch_form"):
        uploaded_file = st.file_uploader("Choose a loan CSV", type=["csv"])
        batch_submitted = st.form_submit_button("Score uploaded loans", type="primary")

    if batch_submitted:
        if uploaded_file is None:
            st.error("Choose a CSV file first.")
        else:
            try:
                batch_data = pd.read_csv(uploaded_file)
                missing_features = [column for column in FEATURES if column not in batch_data.columns]
                if missing_features:
                    st.error(f"CSV is missing required columns: {', '.join(missing_features)}")
                elif batch_data[FEATURES].isna().any().any():
                    st.error("The feature columns contain missing values. Fill them before scoring.")
                else:
                    scored_data = batch_data.copy()
                    scored_features = scored_data[FEATURES]
                    scored_data["predicted.status"] = selected_model.predict(scored_features)
                    scored_data["predicted.status"] = scored_data["predicted.status"].map(
                        {0: "Fully paid", 1: "Not fully paid"}
                    )
                    scored_data["estimated.not.fully.paid.probability"] = selected_model.predict_proba(
                        scored_features
                    )[:, list(selected_model.classes_).index(1)]
                    st.dataframe(scored_data.head(100), use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download predictions as CSV",
                        data=scored_data.to_csv(index=False).encode("utf-8"),
                        file_name="loan_predictions.csv",
                        mime="text/csv",
                    )
                    if len(scored_data) > 100:
                        st.caption(f"Showing the first 100 of {len(scored_data):,} scored rows. The download contains all rows.")
            except (pd.errors.ParserError, UnicodeDecodeError) as error:
                st.error(f"Could not read that CSV: {error}")
