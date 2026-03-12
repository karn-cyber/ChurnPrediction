# Customer Churn Prediction

A production-style machine learning and analytics project built on the Telco Customer Churn dataset.

## Project Structure

- data/Telco-Customer-Churn.csv
- notebooks/churn_analysis.ipynb
- src/data_preprocessing.py
- src/train_model.py
- src/predict.py
- models/churn_model.pkl
- app/streamlit_app.py
- requirements.txt

## What This Project Includes

### ML Pipeline
1. Load dataset
2. Clean `TotalCharges`
3. Remove missing values
4. Encode categorical features with `get_dummies`
5. Train/test split (80/20)
6. Scale numeric features with `StandardScaler`
7. Train Logistic Regression
8. Train Random Forest
9. Compare models (ROC-AUC, Accuracy, Precision, Recall, F1)
10. Save best model to `models/churn_model.pkl`

### Streamlit Dashboard
- Dark, modern SaaS-style design
- KPI overview page
- Data insights with interactive Plotly charts
- Feature importance page
- Customer prediction form with:
  - churn probability
  - risk classification card (Low/Medium/High)
  - gauge visualization
- Business insights and retention recommendations
- Interactive filters and responsive charts

## Run Locally

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Train and save model
```bash
python src/train_model.py --data-path data/Telco-Customer-Churn.csv --model-output models/churn_model.pkl
```

### 3) Launch Streamlit app
```bash
streamlit run app/streamlit_app.py
```

Open the URL shown in your terminal (typically `http://localhost:8501`).

## Streamlit Cloud Deployment
1. Push this project to a GitHub repository.
2. In Streamlit Cloud, create a new app connected to the repository.
3. Set entrypoint to:
   - `app/streamlit_app.py`
4. Ensure `requirements.txt` is at project root.
5. Deploy.

## Notes
- The model artifact stores preprocessing metadata (`feature_columns`, `numeric_columns`, `scaler`) to keep training and inference consistent.
- If the model file is missing, run training first before starting the app.
# ChurnPrediction
