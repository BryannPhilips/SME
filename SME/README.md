# 🇳🇬 Nigerian SME Monthly Sales Prediction

Forecast monthly revenue for Nigerian SMEs using PyCaret AutoML + Streamlit.

---

## 📁 Project Structure

```
sme-sales-prediction/
├── train.py              # AutoML training pipeline
├── app.py                # Streamlit prediction web app
├── requirements.txt      # Python dependencies
├── data/
│   └── dataset.csv       # Training dataset (4,000 SME records)
└── model/
    └── best_model.pkl    # Saved model (generated after training)
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train.py
```
This will:
- Load `data/dataset.csv`
- Auto-detect regression task
- Compare all PyCaret models
- Tune the best model
- Save to `model/best_model.pkl`

### 3. Launch the web app
```bash
streamlit run app.py
```
Visit `http://localhost:8501`

---

## 🚀 Deploy to Streamlit Cloud

```bash
git init && git add . && git commit -m "SME predictor"
git remote add origin https://github.com/YOUR_USERNAME/sme-sales-prediction.git
git push -u origin main
```

Then go to [share.streamlit.io](https://share.streamlit.io), connect your repo, set entry point to `app.py`.

> **Note:** Make sure `model/best_model.pkl` is committed to GitHub, or add model training to the app startup.

---

## 📊 Dataset

| Feature | Type | Description |
|---|---|---|
| business_type | Categorical | Retail, Restaurant, Salon, etc. |
| location_type | Categorical | Market, Mall, Street, Estate, Online |
| state | Categorical | Lagos, Abuja, Kano, etc. |
| num_employees | Integer | 1–20 |
| marketing_spend_naira | Integer | ₦5K–₦200K |
| inventory_value_naira | Integer | ₦100K–₦5M |
| **monthly_sales_thousands** | **Float** | **Target: revenue in ₦K** |

---

## 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Model not found | Run `python train.py` first |
| PyCaret version conflict | Use Python 3.8–3.10 |
| Slow training | Normal — PyCaret compares ~18 models |
