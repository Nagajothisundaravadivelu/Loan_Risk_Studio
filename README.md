# Loan Risk Studio

A Streamlit coursework app for the LendingClub loan dataset. It trains a decision tree and random forest from `loan_data.csv`, compares holdout metrics, predicts one loan from a form, and supports batch scoring from an uploaded CSV.

## Run on Windows

Open PowerShell in this folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Keep `loan_data.csv`, `app.py`, and `requirements.txt` in this folder. Streamlit will print a local URL, usually `http://localhost:8501`.

For batch predictions, upload a CSV containing the same feature columns as `loan_data.csv`. The `not.fully.paid` target column is optional and is ignored by the predictor. Download the scored rows from the app.

**Educational use only:** the sample is class-imbalanced and the models are not validated for real lending or credit decisions.
