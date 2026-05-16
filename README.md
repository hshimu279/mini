

````markdown
# 🌫️ Real-Time Indian Air Quality Index + ML Prediction System

A Streamlit-based **real-time air quality monitoring and prediction system** built using data from [data.gov.in (CPCB)](https://data.gov.in/), integrated with a machine learning module to train and predict AQI categories or numerical values.

---

## 🧠 Features

### 🔹 Real-Time Dashboard
- Fetches **live AQI data** from the **data.gov.in API**.
- Automatically detects correct AQI columns (`pollutant_avg`, `avg_value`, etc.).
- Visualizes station-wise AQI levels in a **Pie Chart**.
- Displays color-coded AQI categories: `Good`, `Satisfactory`, `Moderate`, `Poor`, `Very Poor`, `Severe`.
- Provides **personalized health advice** and **precautions** based on real-time AQI levels.
- Generates a **health risk score** (0–100 scale) for clear interpretation.

### 🔹 ML Model Trainer
- Upload your own dataset (CSV) for model training.
- Automatically detects numeric or categorical AQI targets.
- Performs data preprocessing: Imputation, Encoding, Scaling.
- Trains using **Random Forest Regressor/Classifier**.
- Provides evaluation metrics:
  - `MAE`, `RMSE`, `R²` for regression
  - `Accuracy`, `Precision`, `Recall`, `F1-score` for classification
- Offers **downloadable trained model (.joblib)**.
- Generates conclusions and health insights for predicted AQI.

---

## ⚙️ Tech Stack

| Component | Technology |
|------------|-------------|
| Frontend | Streamlit |
| Backend | Python 3.9+ |
| ML | Scikit-learn, NumPy, Pandas |
| Visualization | Plotly, Matplotlib |
| API | [data.gov.in - CPCB Real-Time Air Quality API](https://data.gov.in/resources/cpcb-real-time-air-quality-data) |
| Environment | `.env` for storing API keys securely |

---

## 🧩 Environment Setup

### 1️⃣ Clone the Repository
```bash
git clone  https://github.com/kanika-manwal/-Real-time-air-quality-ml-system-
cd <your-repo-name>
````

### 2️⃣ Create a Virtual Environment

```bash
python -m venv .a
.a\Scripts\activate    # For Windows
# OR
source .a/bin/activate  # For Linux/Mac
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Create `.env` File

Create a `.env` file in your project folder with:

```
API_KEY=your_data_gov_in_api_key
RESOURCE_ID=95890cbe-4f81-4dd8-bb15-48fe4cc1fe8a
```

You can find your API key from:
👉 [https://data.gov.in/resources/cpcb-real-time-air-quality-data](https://data.gov.in/resources/cpcb-real-time-air-quality-data)

---

## ▶️ Run the App

```bash
streamlit run air_quality_train_test_app.py
```

Then open your browser at:
📍 [http://localhost:8501](http://localhost:8501)

---

## 📊 Example Outputs

* **Live AQI Dashboard**

  * Shows real-time station data with AQI categories and health tips.
* **ML Mode**

  * Trains and evaluates models for AQI prediction.
  * Visualizes predictions and allows model downloads.

---

## ⭐️ Show Your Support

If you found this project helpful, don’t forget to ⭐️ **star the repo** on GitHub!

````

---

## 📦 Create `requirements.txt`
If you don’t already have one, run:
```bash
pip freeze > requirements.txt
````

Typical dependencies:

```
streamlit
pandas
numpy
scikit-learn
plotly
joblib
python-dotenv
requests
matplotlib
```

---

## 🚀 GitHub Push Commands (Step-by-Step)

Run these commands **inside your project folder**:

```bash
# 1️⃣ Initialize Git repo
git init

# 2️⃣ Add all project files
git add .

# 3️⃣ Commit changes
git commit -m "Initial commit: Real-Time Air Quality + ML Prediction System"

# 4️⃣ Link to your GitHub repository
git remote add origin https://github.com/<your-username>/<your-repo-name>.git

# 5️⃣ Push your code to GitHub
git branch -M main
git push -u origin main
```


Would you like me to generate a **shorter version of the README (for GitHub description section)** and a **project tagline** (for your GitHub repo header)?
