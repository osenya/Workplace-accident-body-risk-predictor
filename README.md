# Workplace Accident Risk Prediction

A machine learning project that predicts the **part of the body most likely to be affected** in a workplace accident, based on historical incident data. The trained model is served through an interactive Streamlit dashboard that turns raw accident records into actionable safety insights.

---

## Project Structure

```
ML_PartofBodyAfected/
├── app.py                          # Streamlit dashboard (main entry point)
├── Data_cleansing.py               # Reusable data cleaning pipeline
├── ML.ipynb                        # Model training & evaluation notebook
├── waze.ipynb                      # Supporting / exploratory notebook
├── spss_revised_data.csv           # Primary cleaned dataset
├── spss revised data.xlsx          # Original SPSS export
├── spss revised data 21.10.25.xlsx # Versioned SPSS export (Oct 2025)
├── scaler.pkl                      # Fitted StandardScaler
├── label_encoder.pkl               # General label encoder
├── label_encoder_Partofbodyaffected.pkl  # Target label encoder
└── models/
    ├── best_model_Partofbodyaffected_logistic_regression.pkl  # Trained model
    └── training_columns_Partofbodyaffected.pkl                # Feature column list
```

---

## How It Works

1. **Data Cleaning** (`Data_cleansing.py`) — A standalone pipeline that handles missing values (drop or impute), outlier removal (IQR or Z-score), and duplicate detection. Run it directly or import its functions into other scripts.

2. **Model Training** (`ML.ipynb`) — The notebook loads the cleaned data, encodes features, scales inputs, trains a Logistic Regression classifier, and serialises all artifacts (model, scaler, encoders, column list) to `.pkl` files.

3. **Dashboard** (`app.py`) — A Streamlit app that loads the saved artifacts, accepts a new CSV upload, generates predictions, and renders three visualisations: a ranked bar chart of top risk areas, an injury distribution donut chart, and a Pareto (80/20) analysis. Results can be downloaded as a CSV.

---

## Setup

### Prerequisites

- Python 3.9 or later
- pip

### Install dependencies

```bash
pip install streamlit pandas numpy scikit-learn joblib plotly openpyxl
```

### Run the dashboard

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`), upload a CSV file via the sidebar, and the dashboard will display predictions instantly.

---

## Using the Data Cleaning Pipeline

```python
from Data_cleansing import clean_pipeline

cleaned_df = clean_pipeline(
    filepath="raw_accidents.csv",
    output_path="cleaned_accidents.csv",
    missing_strategy="auto",   # 'drop' | 'impute' | 'auto'
    outlier_method="iqr",      # 'iqr'  | 'zscore'
)
```

Or use individual functions for more control:

```python
from Data_cleansing import load_data, handle_missing_values, remove_outliers, remove_duplicates

df = load_data("raw_accidents.csv")
df = handle_missing_values(df, strategy="impute")
df = remove_outliers(df, method="zscore", z_thresh=3.0)
df = remove_duplicates(df)
```

---

## Input CSV Format

The uploaded CSV should match the schema used during training (same column names). The app automatically aligns columns to the training feature set and fills any missing columns with zeros.

---

## Model Details

| Item | Detail |
|---|---|
| Algorithm | Logistic Regression |
| Target | Part of body affected |
| Preprocessing | One-hot encoding (categorical), StandardScaler (numeric) |
| Artifacts | `.pkl` files via `joblib` |

---

## Notes

- Predictions are based on historical patterns and should **support — not replace** professional safety judgment.
- To retrain the model, run all cells in `ML.ipynb` and the new `.pkl` files will overwrite the existing ones.
