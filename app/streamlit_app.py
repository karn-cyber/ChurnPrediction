from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
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


@st.cache_data
def build_analysis_frame(df: pd.DataFrame) -> pd.DataFrame:
    analysis = df.copy()
    analysis["ChurnFlag"] = (analysis["Churn"] == "Yes").astype(int)
    analysis["ContractRank"] = analysis["Contract"].map({"Month-to-month": 0, "One year": 1, "Two year": 2}).fillna(0)
    analysis["ChargePerTenure"] = analysis["TotalCharges"] / analysis["tenure"].replace(0, np.nan)
    analysis["ChargePerTenure"] = analysis["ChargePerTenure"].fillna(analysis["MonthlyCharges"])
    return analysis


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
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1280px;
        }
        .kpi-card, .insight-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            box-shadow: 0 6px 20px rgba(0,0,0,0.25);
            margin-bottom: 0.5rem;
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
        .step-title {
            color: #dbeafe;
            font-weight: 700;
            margin-bottom: 0.35rem;
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


def insight_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class='insight-card'>
            <div class='step-title'>{title}</div>
            <div>{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_label(prob: float) -> tuple[str, str]:
    if prob < 0.35:
        return "Low Risk", "#22c55e"
    if prob < 0.65:
        return "Medium Risk", "#eab308"
    return "High Risk", "#ef4444"


def make_gauge(probability: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"color": "#f9fafb", "size": 34}},
            title={"text": "Predicted Churn Probability", "font": {"color": "#e5e7eb", "size": 18}},
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


def make_numeric_correlation_heatmap(analysis_df: pd.DataFrame) -> go.Figure:
    numeric_cols = ["ChurnFlag", "tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "ContractRank", "ChargePerTenure"]
    corr = analysis_df[numeric_cols].corr().round(2)
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            colorbar={"title": "Correlation"},
            text=corr.values,
            texttemplate="%{text}",
        )
    )
    fig.update_layout(title="Numeric Correlation Heatmap", template="plotly_dark", height=520)
    return fig


def make_categorical_churn_heatmap(df: pd.DataFrame) -> go.Figure:
    grid = (
        df.groupby(["Contract", "PaymentMethod"], as_index=False)["Churn"]
        .apply(lambda s: (s == "Yes").mean() * 100)
        .rename(columns={"Churn": "ChurnRate"})
    )
    pivot = grid.pivot(index="PaymentMethod", columns="Contract", values="ChurnRate").fillna(0).round(1)

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale="YlOrRd",
            colorbar={"title": "Churn %"},
            text=pivot.values,
            texttemplate="%{text}%",
        )
    )
    fig.update_layout(title="Churn Rate Heatmap: Contract x Payment Method", template="plotly_dark", height=450)
    return fig


def overview_page(df: pd.DataFrame) -> None:
    st.subheader(":material/insights: Overview")
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
        kpi_card("Average Monthly Charges", f"${avg_monthly:,.2f}")
    with c4:
        kpi_card("Average Tenure", f"{avg_tenure:.1f} months")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(df, names="Churn", title="Customer Churn Distribution", template="plotly_dark", hole=0.5)
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        segment = df.groupby("Contract", as_index=False)["customerID"].count().rename(columns={"customerID": "Customers"})
        fig = px.bar(segment, x="Contract", y="Customers", title="Customers by Contract Type", color="Contract", template="plotly_dark")
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)

    insight_card(
        "Library & Process",
        "Charts above use Plotly Express. Process: aggregate customer counts and churn labels, then visualize distribution for fast segmentation decisions.",
    )
    insight_card(
        "Conclusion & Implication",
        "If month-to-month volume and churn share are both high, retention actions should prioritize contract migration campaigns and early lifecycle interventions.",
    )


def insights_page(df: pd.DataFrame) -> None:
    st.subheader(":material/analytics: Data Explorer")

    with st.expander("Interactive Filters", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            selected_contracts = st.multiselect("Contract Type", sorted(df["Contract"].unique()), default=sorted(df["Contract"].unique()))
        with c2:
            selected_internet = st.multiselect("Internet Service", sorted(df["InternetService"].unique()), default=sorted(df["InternetService"].unique()))
        with c3:
            selected_gender = st.multiselect("Gender", sorted(df["gender"].unique()), default=sorted(df["gender"].unique()))
        with c4:
            tenure_range = st.slider("Tenure Range", int(df["tenure"].min()), int(df["tenure"].max()), (int(df["tenure"].min()), int(df["tenure"].max())))

    filtered = df[
        (df["Contract"].isin(selected_contracts))
        & (df["InternetService"].isin(selected_internet))
        & (df["gender"].isin(selected_gender))
        & (df["tenure"].between(tenure_range[0], tenure_range[1]))
    ]

    c1, c2 = st.columns(2)
    with c1:
        churn_dist = filtered["Churn"].value_counts().reset_index()
        churn_dist.columns = ["Churn", "Count"]
        fig = px.bar(churn_dist, x="Churn", y="Count", color="Churn", title="Churn Distribution", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(filtered, x="tenure", color="Churn", barmode="overlay", nbins=30, title="Tenure Distribution by Churn", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        contract_churn = filtered.groupby(["Contract", "Churn"], as_index=False).size().rename(columns={"size": "Count"})
        fig = px.bar(contract_churn, x="Contract", y="Count", color="Churn", barmode="group", title="Contract vs Churn", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = px.box(filtered, x="Churn", y="MonthlyCharges", color="Churn", title="Monthly Charges by Churn", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        fig = px.scatter(
            filtered,
            x="MonthlyCharges",
            y="TotalCharges",
            color="Churn",
            size="tenure",
            title="Charges Relationship (size = tenure)",
            template="plotly_dark",
            hover_data=["Contract", "InternetService", "PaymentMethod"],
        )
        st.plotly_chart(fig, use_container_width=True)
    with c6:
        fig = px.violin(filtered, x="Churn", y="tenure", color="Churn", box=True, title="Tenure vs Churn", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    insight_card(
        "Library & Process",
        "All explorer visuals use Plotly. Process: filtered slicing + grouped aggregations + distribution charts to isolate high-risk segments by contract, tenure, and billing behavior.",
    )
    insight_card(
        "Conclusion & Implication",
        "When churned users cluster at low tenure with higher monthly charges, onboarding quality and price-value communication become immediate optimization levers.",
    )


def correlation_page(df: pd.DataFrame, analysis_df: pd.DataFrame) -> None:
    st.subheader(":material/grid_view: Correlation & Heatmaps")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(make_numeric_correlation_heatmap(analysis_df), use_container_width=True)
    with col2:
        st.plotly_chart(make_categorical_churn_heatmap(df), use_container_width=True)

    churn_by_tenure_bucket = (
        analysis_df.assign(TenureBucket=pd.cut(analysis_df["tenure"], bins=[0, 12, 24, 36, 48, 60, 72], include_lowest=True))
        .groupby("TenureBucket", as_index=False, observed=False)["ChurnFlag"]
        .mean()
    )
    churn_by_tenure_bucket["TenureBucket"] = churn_by_tenure_bucket["TenureBucket"].map(lambda x: str(x))
    churn_by_tenure_bucket["ChurnRate"] = churn_by_tenure_bucket["ChurnFlag"] * 100

    fig = go.Figure(
        data=[
            go.Scatter(
                x=[str(v) for v in churn_by_tenure_bucket["TenureBucket"].tolist()],
                y=[float(v) for v in churn_by_tenure_bucket["ChurnRate"].tolist()],
                mode="lines+markers",
                line={"color": "#60a5fa", "width": 3},
                marker={"size": 9},
                name="ChurnRate",
            )
        ]
    )
    fig.update_layout(
        title="Churn Rate Trend Across Tenure Buckets",
        template="plotly_dark",
        xaxis_title="Tenure Bucket",
        yaxis_title="Churn Rate (%)",
    )
    st.plotly_chart(fig, use_container_width=True)

    insight_card(
        "Library & Process",
        "Correlations are computed using pandas `.corr()` and rendered through Plotly Heatmap. Categorical churn heatmap is built from grouped churn-rate percentages.",
    )
    insight_card(
        "Conclusion & Implication",
        "Positive correlation between churn and monthly charges (or weak tenure) indicates pricing pressure and early lifecycle risk. Contract-payment intersections reveal where targeted retention should be prioritized.",
    )


def model_lab_page(artifact: dict, analysis_df: pd.DataFrame) -> None:
    st.subheader(":material/model_training: Model Lab")

    st.markdown("#### Model Comparison")
    metric_rows: list[dict] = []
    for model_name, metrics in artifact["metrics"].items():
        row = {"Model": model_name}
        row.update({k.upper(): round(v, 4) for k, v in metrics.items()})
        metric_rows.append(row)
    metric_df = pd.DataFrame(metric_rows).sort_values("ROC_AUC", ascending=False)
    st.dataframe(metric_df, use_container_width=True, hide_index=True)

    comp_fig = px.bar(metric_df, x="Model", y="ROC_AUC", color="Model", title="ROC-AUC Comparison", template="plotly_dark")
    st.plotly_chart(comp_fig, use_container_width=True)

    model = artifact["model"]
    if hasattr(model, "feature_importances_"):
        fi_df = pd.DataFrame({"Feature": artifact["feature_columns"], "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
        top_n = st.slider("Top Feature Count", min_value=5, max_value=35, value=20, step=1)
        fig = px.bar(
            fi_df.head(top_n).sort_values("Importance", ascending=True),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Random Forest Feature Importance",
            template="plotly_dark",
        )
        fig.update_layout(height=650)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Best model is Logistic Regression. Feature importances are not directly available from tree-based gain metrics.")

    sample = analysis_df.sample(min(300, len(analysis_df)), random_state=42)
    fig = px.scatter(
        sample,
        x="tenure",
        y="MonthlyCharges",
        color="Churn",
        title="Decision Surface Proxy: tenure vs monthly charges",
        template="plotly_dark",
        hover_data=["Contract", "PaymentMethod", "InternetService"],
    )
    st.plotly_chart(fig, use_container_width=True)

    insight_card(
        "Library & Process",
        "Model training uses scikit-learn (Logistic Regression + Random Forest). Selection process compares ROC-AUC, Accuracy, Precision, Recall, and F1 on a holdout test split.",
    )
    insight_card(
        "What the Model is Doing",
        "The model learns probability of churn from behavioral and billing patterns. It weighs encoded customer attributes and outputs a churn likelihood score used for risk segmentation.",
    )
    insight_card(
        "Conclusion & Implication",
        "High-importance features should guide intervention strategy. If pricing, tenure, and contract-related fields dominate, retention spend should focus on tenure acceleration and contract conversion.",
    )


def prediction_page(predictor: ChurnPredictor) -> None:
    st.subheader(":material/person_search: Customer Prediction")

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
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            )

        submitted = st.form_submit_button("Run Prediction")

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

        with st.spinner("Running inference pipeline..."):
            prob = predictor.predict_proba(payload)

        label, color = risk_label(prob)
        st.plotly_chart(make_gauge(prob), use_container_width=True)
        st.markdown(
            f"""
            <div class='result-card'>
                <h3 style='margin:0;color:{color};'>Risk Level: {label}</h3>
                <p style='margin:0.35rem 0 0 0;'>Predicted churn probability: <b>{prob:.2%}</b></p>
                <p style='margin:0.35rem 0 0 0;'>Inference stack: pandas preprocessing + one-hot encoding + scaler transform + trained scikit-learn classifier.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def business_insights_page(df: pd.DataFrame) -> None:
    st.subheader(":material/lightbulb: Business Insights")

    churn_rate = (df["Churn"] == "Yes").mean() * 100
    month_to_month_churn = ((df[df["Contract"] == "Month-to-month"]["Churn"] == "Yes").mean()) * 100
    electronic_check_churn = ((df[df["PaymentMethod"] == "Electronic check"]["Churn"] == "Yes").mean()) * 100

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Global Churn Rate", f"{churn_rate:.1f}%")
    with c2:
        st.metric("Month-to-Month Churn Rate", f"{month_to_month_churn:.1f}%")
    with c3:
        st.metric("Electronic Check Churn Rate", f"{electronic_check_churn:.1f}%")

    retention_df = (
        df.groupby(["Contract", "PaymentMethod"], as_index=False)["Churn"]
        .apply(lambda s: (s == "Yes").mean() * 100)
        .rename(columns={"Churn": "ChurnRate"})
        .sort_values("ChurnRate", ascending=False)
        .head(12)
    )

    fig = px.bar(
        retention_df,
        x="ChurnRate",
        y="PaymentMethod",
        color="Contract",
        orientation="h",
        title="Highest Risk Contract/Payment Segments",
        template="plotly_dark",
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    insight_card("Recommendation 1", "Design targeted migration offers from month-to-month to annual contracts with first-cycle incentives.")
    insight_card("Recommendation 2", "Trigger intervention for customers with high monthly charge growth in first 12 months.")
    insight_card("Recommendation 3", "Prioritize outreach workflows for high-risk payment methods and low-tenure segments.")


def methodology_page() -> None:
    st.subheader(":material/menu_book: Methodology, Libraries, and Interpretation")

    steps = [
        ("1) Data Loading", "Library: pandas", "`read_csv` loads Telco customer records into a dataframe.", "Implication: all downstream analytics depend on schema integrity."),
        ("2) Data Cleaning", "Library: pandas", "`to_numeric(errors='coerce')` cleans `TotalCharges`; rows with missing values are dropped.", "Implication: removes invalid billing records to prevent model bias from malformed data."),
        ("3) Feature Encoding", "Library: pandas", "`get_dummies` converts categorical columns to binary indicators.", "Implication: enables machine learning models to process non-numeric customer attributes."),
        ("4) Train/Test Split", "Library: scikit-learn", "`train_test_split(..., test_size=0.2, stratify=y)` creates evaluation holdout data.", "Implication: provides realistic estimate of model generalization performance."),
        ("5) Feature Scaling", "Library: scikit-learn", "`StandardScaler` standardizes numeric features.", "Implication: keeps model coefficients and distance-sensitive learning numerically stable."),
        ("6) Model Training", "Library: scikit-learn", "Trains Logistic Regression and Random Forest classifiers.", "Implication: compares linear and non-linear decision boundaries for churn prediction."),
        ("7) Model Evaluation", "Library: scikit-learn", "Computes Accuracy, Precision, Recall, F1, and ROC-AUC.", "Implication: balances false positives/negatives and overall ranking quality."),
        ("8) Model Selection", "Library: Python + scikit-learn", "Best model selected using ROC-AUC from holdout metrics.", "Implication: selected model is optimized for probability ranking quality."),
        ("9) Serialization", "Library: joblib", "Saves model, scaler, and feature schema into `churn_model.pkl`.", "Implication: guarantees consistent inference in production."),
        ("10) Dashboard Analytics", "Library: Streamlit + Plotly", "Interactive KPI cards, filters, charts, heatmaps, and prediction workflow.", "Implication: turns model output into actionable retention decisions for business users."),
    ]

    for step, lib, process, implication in steps:
        st.markdown(f"#### {step}")
        st.markdown(f"**{lib}**")
        st.markdown(f"- Process: {process}")
        st.markdown(f"- Meaning: {implication}")


def data_table_page(df: pd.DataFrame) -> None:
    st.subheader(":material/table_view: Data Catalog")
    st.caption("Use this section to inspect complete filtered data and download records for external analysis.")

    col1, col2 = st.columns(2)
    with col1:
        churn_filter = st.multiselect("Churn Filter", sorted(df["Churn"].unique()), default=sorted(df["Churn"].unique()))
    with col2:
        contract_filter = st.multiselect("Contract Filter", sorted(df["Contract"].unique()), default=sorted(df["Contract"].unique()))

    filtered = df[df["Churn"].isin(churn_filter) & df["Contract"].isin(contract_filter)]
    st.dataframe(filtered, use_container_width=True, height=420)

    csv_data = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download Filtered Data (CSV)", data=csv_data, file_name="filtered_churn_data.csv", mime="text/csv")


def main() -> None:
    st.set_page_config(page_title="Customer Churn Analytics", page_icon=":material/monitoring:", layout="wide")
    inject_custom_css()

    st.title(":material/dashboard: Customer Churn Intelligence Platform")
    st.caption("Interactive analytics, model transparency, and customer risk scoring in one production dashboard")

    if not DATA_PATH.exists():
        st.error(f"Dataset not found at {DATA_PATH}")
        st.stop()
    if not MODEL_PATH.exists():
        st.error("Model not found. Train first using src/train_model.py")
        st.stop()

    df = load_data()
    analysis_df = build_analysis_frame(df)
    predictor = load_predictor()
    artifact = load_model_artifact()

    page = st.sidebar.radio(
        "Navigation",
        [
            "Overview",
            "Data Explorer",
            "Correlation & Heatmaps",
            "Model Lab",
            "Customer Prediction",
            "Business Insights",
            "Methodology",
            "Data Catalog",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.info(f"Model in production: {artifact.get('model_name', 'N/A')}")
    st.sidebar.caption("Libraries: pandas, scikit-learn, Streamlit, Plotly, joblib")

    if page == "Overview":
        overview_page(df)
    elif page == "Data Explorer":
        insights_page(df)
    elif page == "Correlation & Heatmaps":
        correlation_page(df, analysis_df)
    elif page == "Model Lab":
        model_lab_page(artifact, analysis_df)
    elif page == "Customer Prediction":
        prediction_page(predictor)
    elif page == "Business Insights":
        business_insights_page(df)
    elif page == "Methodology":
        methodology_page()
    else:
        data_table_page(df)


if __name__ == "__main__":
    main()
