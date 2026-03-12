from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.predict import ChurnPredictor  # noqa: E402

DATA_PATH = PROJECT_ROOT / "data" / "Telco-Customer-Churn.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.pkl"


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    return df


@st.cache_resource
def load_predictor() -> ChurnPredictor:
    return ChurnPredictor(MODEL_PATH)


@st.cache_resource
def load_model_artifact() -> dict:
    return joblib.load(MODEL_PATH)


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(120deg, #0f172a 0%, #111827 45%, #0b1120 100%);
            color: #f3f4f6;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        .kpi-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            box-shadow: 0 6px 20px rgba(0,0,0,0.25);
        }
        .kpi-title {
            color: #9ca3af;
            font-size: 0.85rem;
            margin-bottom: 0.2rem;
        }
        .kpi-value {
            color: #f9fafb;
            font-size: 1.6rem;
            font-weight: 700;
        }
        .result-card {
            border-radius: 14px;
            padding: 1rem 1.2rem;
            margin-top: 1rem;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(title: str, value: str) -> None:
    st.markdown(
        f"""
        <div class='kpi-card'>
            <div class='kpi-title'>{title}</div>
            <div class='kpi-value'>{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_label(prob: float) -> tuple[str, str]:
    if prob < 0.35:
        return "🟢 Low Risk", "#22c55e"
    if prob < 0.65:
        return "🟡 Medium Risk", "#eab308"
    return "🔴 High Risk", "#ef4444"


def make_gauge(probability: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"color": "#f9fafb", "size": 34}},
            title={"text": "Churn Risk Probability", "font": {"color": "#e5e7eb", "size": 18}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#9ca3af"},
                "bar": {"color": "#60a5fa"},
                "steps": [
                    {"range": [0, 35], "color": "rgba(34,197,94,0.35)"},
                    {"range": [35, 65], "color": "rgba(234,179,8,0.35)"},
                    {"range": [65, 100], "color": "rgba(239,68,68,0.35)"},
                ],
            },
        )
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=320)
    return fig


def overview_page(df: pd.DataFrame) -> None:
    st.subheader("Overview")
    total_customers = len(df)
    churn_rate = (df["Churn"].eq("Yes").mean()) * 100
    avg_monthly = df["MonthlyCharges"].mean()
    avg_tenure = df["tenure"].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Customers", f"{total_customers:,}")
    with c2:
        kpi_card("Churn Rate", f"{churn_rate:.2f}%")
    with c3:
        kpi_card("Avg Monthly Charges", f"${avg_monthly:,.2f}")
    with c4:
        kpi_card("Avg Tenure", f"{avg_tenure:.1f} months")

    st.markdown("### Customer Mix")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(df, names="Churn", title="Churn Distribution", template="plotly_dark", hole=0.45)
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        segment = (
            df.groupby("Contract", as_index=False)["customerID"]
            .count()
            .rename(columns={"customerID": "Customers"})
        )
        fig = px.bar(
            segment,
            x="Contract",
            y="Customers",
            title="Customers by Contract Type",
            color="Contract",
            template="plotly_dark",
        )
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)


def insights_page(df: pd.DataFrame) -> None:
    st.subheader("Data Insights")

    with st.expander("Filters", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            selected_contracts = st.multiselect(
                "Contract Type", options=sorted(df["Contract"].unique()), default=sorted(df["Contract"].unique())
            )
        with c2:
            selected_internet = st.multiselect(
                "Internet Service",
                options=sorted(df["InternetService"].unique()),
                default=sorted(df["InternetService"].unique()),
            )
        with c3:
            selected_gender = st.multiselect(
                "Gender", options=sorted(df["gender"].unique()), default=sorted(df["gender"].unique())
            )

    filtered = df[
        (df["Contract"].isin(selected_contracts))
        & (df["InternetService"].isin(selected_internet))
        & (df["gender"].isin(selected_gender))
    ]

    c1, c2 = st.columns(2)
    with c1:
        churn_dist = filtered["Churn"].value_counts().reset_index()
        churn_dist.columns = ["Churn", "Count"]
        fig = px.bar(churn_dist, x="Churn", y="Count", color="Churn", title="Churn Distribution", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        contract_churn = (
            filtered.groupby(["Contract", "Churn"], as_index=False)
            .size()
            .rename(columns={"size": "Count"})
        )
        fig = px.bar(
            contract_churn,
            x="Contract",
            y="Count",
            color="Churn",
            barmode="group",
            title="Contract vs Churn",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.box(
            filtered,
            x="Churn",
            y="tenure",
            color="Churn",
            title="Tenure vs Churn",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        fig = px.violin(
            filtered,
            x="Churn",
            y="MonthlyCharges",
            color="Churn",
            box=True,
            title="Monthly Charges vs Churn",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)


def feature_importance_page(artifact: dict) -> None:
    st.subheader("Feature Importance")
    model = artifact["model"]

    if not hasattr(model, "feature_importances_"):
        st.info("Feature importance is available when Random Forest is selected as best model.")
        return

    feature_importance_df = pd.DataFrame(
        {
            "Feature": artifact["feature_columns"],
            "Importance": model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    top_n = st.slider("Top features", min_value=5, max_value=30, value=15, step=1)
    fig = px.bar(
        feature_importance_df.head(top_n).sort_values("Importance", ascending=True),
        x="Importance",
        y="Feature",
        orientation="h",
        title=f"Top {top_n} Feature Importances",
        template="plotly_dark",
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)


def prediction_page(predictor: ChurnPredictor) -> None:
    st.subheader("Customer Prediction")

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", [0, 1])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        with c2:
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            monthly = st.slider("Monthly Charges", 15.0, 130.0, 75.0)
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

        submitted = st.form_submit_button("Predict Churn")

    if submitted:
        payload = {
            "gender": gender,
            "SeniorCitizen": senior,
            "tenure": tenure,
            "Contract": contract,
            "InternetService": internet,
            "MonthlyCharges": monthly,
            "PaymentMethod": payment,
            "TotalCharges": float(tenure * monthly),
        }

        with st.spinner("Scoring customer risk..."):
            prob = predictor.predict_proba(payload)

        label, color = risk_label(prob)
        st.plotly_chart(make_gauge(prob), use_container_width=True)

        st.markdown(
            f"""
            <div class='result-card'>
                <h3 style='margin:0;color:{color};'>{label}</h3>
                <p style='margin:0.35rem 0 0 0;'>Estimated churn probability: <b>{prob:.2%}</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def business_insights_page(df: pd.DataFrame) -> None:
    st.subheader("Business Insights")

    high_risk = df[(df["Contract"] == "Month-to-month") & (df["MonthlyCharges"] > df["MonthlyCharges"].median())]

    st.markdown("### High Churn Customer Characteristics")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Month-to-month in high-risk set", f"{(high_risk['Contract'] == 'Month-to-month').mean() * 100:.1f}%")
    with c2:
        st.metric("Avg Monthly Charges (high-risk)", f"${high_risk['MonthlyCharges'].mean():.2f}")
    with c3:
        st.metric("Avg Tenure (high-risk)", f"{high_risk['tenure'].mean():.1f} months")

    st.markdown("### Retention Recommendations")
    st.markdown(
        """
        - Offer discounted annual plans for month-to-month subscribers.
        - Trigger proactive outreach when monthly charges increase rapidly.
        - Bundle value-added services (Tech Support / Security) for short-tenure users.
        - Build payment-method migration campaigns away from electronic check.
        """
    )

    recommend_df = (
        df.groupby("PaymentMethod", as_index=False)["Churn"]
        .apply(lambda s: (s == "Yes").mean() * 100)
        .rename(columns={"Churn": "ChurnRate"})
        .sort_values("ChurnRate", ascending=False)
    )

    fig = px.bar(
        recommend_df,
        x="PaymentMethod",
        y="ChurnRate",
        color="ChurnRate",
        title="Churn Rate by Payment Method",
        template="plotly_dark",
        color_continuous_scale="Bluered",
    )
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Customer Churn Analytics", page_icon="📊", layout="wide")
    inject_custom_css()

    st.title("📊 Customer Churn Prediction Analytics")
    st.caption("Production-style SaaS dashboard for churn monitoring and risk scoring")

    if not DATA_PATH.exists():
        st.error(f"Dataset not found at {DATA_PATH}")
        st.stop()

    if not MODEL_PATH.exists():
        st.error("Model not found. Train the model first: python src/train_model.py")
        st.stop()

    df = load_data()
    predictor = load_predictor()
    artifact = load_model_artifact()

    page = st.sidebar.radio(
        "Navigate",
        [
            "Overview",
            "Data Insights",
            "Feature Importance",
            "Customer Prediction",
            "Business Insights",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.success(f"Best model: {artifact.get('model_name', 'N/A')}")

    if page == "Overview":
        overview_page(df)
    elif page == "Data Insights":
        insights_page(df)
    elif page == "Feature Importance":
        feature_importance_page(artifact)
    elif page == "Customer Prediction":
        prediction_page(predictor)
    else:
        business_insights_page(df)


if __name__ == "__main__":
    main()
