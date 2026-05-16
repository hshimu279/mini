
import os, io, requests
from dotenv import load_dotenv
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, classification_report
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


load_dotenv()
API_KEY = os.getenv("API_KEY")
RESOURCE_ID = os.getenv("RESOURCE_ID")

AQI_BANDS = [
    (0, 50, "Good", "#009865"),
    (51, 100, "Satisfactory", "#8BC34A"),
    (101, 200, "Moderate", "#FFC107"),
    (201, 300, "Poor", "#FF9800"),
    (301, 400, "Very Poor", "#F44336"),
    (401, 500, "Severe", "#7E0023"),
]
CATEGORY_TO_MID = {name: (lo + hi) / 2 for (lo, hi, name, _) in AQI_BANDS}

HEALTH_ADVICE = {
    "Good": ["Air quality is satisfactory; pollution poses little risk.", "Enjoy outdoor activities."],
    "Satisfactory": ["Mild breathing discomfort for sensitive groups.", "Continue normal activities."],
    "Moderate": ["Limit prolonged outdoor exertion.", "Sensitive people should wear masks outdoors."],
    "Poor": ["Avoid outdoor activity; use air purifiers.", "Masks recommended in polluted areas."],
    "Very Poor": ["Stay indoors; avoid exertion.", "Follow local advisories for pollution."],
    "Severe": ["Hazardous air quality. Avoid outdoor exposure.", "Seek medical help if unwell."],
}

def aqi_to_category(aqi: float):
    try:
        aqi = float(aqi)
    except Exception:
        return "Unknown", "#9E9E9E"
    for lo, hi, cat, color in AQI_BANDS:
        if lo <= aqi <= hi:
            return cat, color
    return "Severe", AQI_BANDS[-1][3]

def safe_avg_from_predictions(y_pred):
    """Convert numeric or categorical predictions to average AQI."""
    try:
        return float(np.mean(y_pred.astype(float))), "numeric"
    except:
        try:
            mapped = [CATEGORY_TO_MID.get(str(x).title(), np.nan) for x in y_pred]
            if np.isnan(mapped).all():
                raise ValueError
            return float(np.nanmean(mapped)), "category"
        except:
            raise ValueError("Predictions are non-numeric and not AQI categories.")

st.set_page_config(page_title="Real-Time AQI Dashboard", layout="wide")
st.title("🌫️ Real-Time Indian Air Quality + ML Prediction System")
mode = st.sidebar.radio("Select Mode", ["📡 Real-Time AQI Dashboard", "🤖 Train/Test ML Model"])

if mode == "📡 Real-Time AQI Dashboard":
    st.header("🇮🇳 Real-Time Air Quality Monitoring (data.gov.in)")

    if not API_KEY or not RESOURCE_ID:
        st.error("⚠️ Missing API_KEY or RESOURCE_ID in .env file.")
    else:
        with st.spinner("Fetching available cities..."):
            try:
                url_all = f"https://api.data.gov.in/resource/{RESOURCE_ID}?api-key={API_KEY}&format=json&limit=10000"
                res = requests.get(url_all)
                data_all = res.json()

                if "records" in data_all:
                    df_all = pd.DataFrame(data_all["records"])
                    if "city" in df_all.columns:
                        cities = sorted(df_all["city"].dropna().unique().tolist())
                    else:
                        cities = []
                else:
                    cities = []
            except Exception as e:
                st.error(f"Error loading city list: {e}")
                cities = []

        if not cities:
            st.warning("⚠️ Could not load city list from API.")
        else:
            city = st.selectbox("Select City", cities, index=cities.index("Delhi") if "Delhi" in cities else 0)

            if st.button("Fetch Live AQI Data"):
                with st.spinner(f"Fetching AQI data for {city}..."):
                    try:
                        url = f"https://api.data.gov.in/resource/{RESOURCE_ID}?api-key={API_KEY}&format=json&filters[city]={city}"
                        res = requests.get(url)
                        data = res.json()

                        if "records" not in data or not data["records"]:
                            st.warning("⚠️ No records found for this city.")
                        else:
                            df = pd.DataFrame(data["records"])
                            st.subheader(f"📍 City: {city}")
                            st.dataframe(df.head(20), use_container_width=True)

                            # 🔍 Auto-detect AQI-related column
                            possible_cols = [
                                "pollutant_avg", "avg_value",
                                "pollutant_avg_value", "pollutant_mean",
                                "value", "aqi", "aqi_value"
                            ]
                            aqi_col = next((c for c in possible_cols if c in df.columns), None)

                            if not aqi_col:
                                st.error("❌ No AQI column found in dataset (pollutant_avg / avg_value / aqi).")
                            else:
                                st.caption(f"✅ Using column: **{aqi_col}** for AQI calculations.")
                                df[aqi_col] = pd.to_numeric(df[aqi_col], errors="coerce")

                                if df[aqi_col].notna().sum() == 0:
                                    st.warning("⚠️ No numeric AQI values available in this dataset.")
                                else:
                                    avg_aqi = df[aqi_col].mean()
                                    cat, color = aqi_to_category(avg_aqi)

                                    # 🌡️ AQI Summary
                                    st.markdown(
                                        f"""
                                        <div style='background:{color};padding:12px;border-radius:8px;color:white;font-size:20px'>
                                        🌡️ <b>Average AQI in {city}:</b> {avg_aqi:.0f} ({cat})
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )

                                    # 🩺 Health Advice
                                    st.write("### 🩺 Health Recommendations")
                                    for tip in HEALTH_ADVICE.get(cat, ["Monitor updates and stay safe."]):
                                        st.write(f"- {tip}")

                                    # 📘 Conclusion
                                    st.write("### 📘 Conclusion")
                                    conclusions = {
                                        "Good": f"The air quality in {city} is excellent. Safe for all outdoor activities.",
                                        "Satisfactory": f"Air quality is satisfactory. Minor discomfort possible to sensitive groups.",
                                        "Moderate": f"Moderate pollution. Limit long outdoor exposure, especially for elderly or children.",
                                        "Poor": f"Pollution levels are high in {city}. Avoid outdoor activity if possible.",
                                        "Very Poor": f"Very poor air quality detected. Stay indoors as much as possible.",
                                        "Severe": f"⚠️ Hazardous air quality! Stay indoors and use air purifiers or masks."
                                    }
                                    st.info(conclusions.get(cat, "Stay informed and follow pollution advisories."))

                                    # 🧠 Health Risk Prediction
                                    st.write("### 💀 Health Risk Level")
                                    risk_score = min(max((avg_aqi / 500) * 100, 0), 100)
                                    if risk_score < 20:
                                        level, risk_color, desc = "Low", "#00C853", "Air quality is safe."
                                    elif risk_score < 40:
                                        level, risk_color, desc = "Mild", "#AEEA00", "Slight risk for sensitive individuals."
                                    elif risk_score < 60:
                                        level, risk_color, desc = "Moderate", "#FFD600", "Mild discomfort possible."
                                    elif risk_score < 80:
                                        level, risk_color, desc = "High", "#FF6D00", "Unhealthy air. Avoid exertion."
                                    else:
                                        level, risk_color, desc = "Critical", "#D50000", "Severe risk to all individuals."

                                    st.markdown(
                                        f"<div style='background:{risk_color};padding:10px;border-radius:8px;color:white;font-size:18px'>"
                                        f"💀 <b>Health Risk Level:</b> {level} ({risk_score:.0f}/100)</div>",
                                        unsafe_allow_html=True,
                                    )
                                    st.progress(int(risk_score))
                                    st.write(desc)

                                    # 🏭 Station-wise AQI Pie Chart
                                    st.write("### 🏭 Station-wise AQI Distribution (Pie Chart)")
                                    station_avg = df.groupby("station")[aqi_col].mean().dropna()
                                    if not station_avg.empty:
                                        import plotly.express as px
                                        fig = px.pie(
                                            names=station_avg.index,
                                            values=station_avg.values,
                                            title=f"AQI Share by Monitoring Station — {city}",
                                            hole=0.4
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.warning("No numeric station data available for visualization.")
                    except Exception as e:
                        st.error(f"Error loading data: {e}")



else:
    st.header("🤖 Train, Evaluate & Analyze AQI Prediction Model")

    file = st.sidebar.file_uploader("Upload Dataset (CSV)", type=["csv"])
    if not file:
        st.info("Upload a dataset with AQI or pollutant data.")
    else:
        df = pd.read_csv(file)
        st.dataframe(df.head(), use_container_width=True)

        # ------------------ Auto-detect AQI column ------------------
        possible_targets = [c for c in df.columns if "aqi" in c.lower() or "pollutant_avg" in c.lower()]
        target_default = possible_targets[0] if possible_targets else df.columns[-1]
        target = st.sidebar.selectbox("🎯 Target Column (auto-detected)", df.columns, index=list(df.columns).index(target_default))

        # City selection
        if "city" in df.columns:
            city_choice = st.selectbox("Select City for Analysis", ["All Cities"] + sorted(df["city"].dropna().unique().tolist()))
            train_city_only = st.checkbox("🏙️ Train on this city only", value=False)
            if train_city_only and city_choice != "All Cities":
                df = df[df["city"] == city_choice]
                st.success(f"Training restricted to {city_choice}")
        else:
            city_choice, train_city_only = "All Cities", False

        # Detect feature types
        datetime_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
        for col in datetime_cols:
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                df[f"{col}_hour"] = df[col].dt.hour
                df[f"{col}_month"] = df[col].dt.month
            except:
                pass

        exclude_cols = [target] + datetime_cols
        features = st.sidebar.multiselect("🧩 Feature Columns", [c for c in df.columns if c not in exclude_cols],
                                          default=[c for c in df.columns if c not in exclude_cols][:6])
        test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2)

        # ------------------ Train Model ------------------
        if st.button("🚀 Train Model"):
            df = df.dropna(subset=[target])
            X, y = df[features], df[target]

            task = "regression" if pd.api.types.is_numeric_dtype(y) else "classification"
            num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
            cat_cols = [c for c in X.columns if c not in num_cols]

            pre = ColumnTransformer([
                ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
                ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                  ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
            ])

            model = RandomForestRegressor(n_estimators=300, random_state=42) if task == "regression" else RandomForestClassifier(n_estimators=300, random_state=42)
            pipe = Pipeline([("pre", pre), ("model", model)])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            st.success("✅ Model trained successfully.")

            st.subheader("📊 Model Evaluation")
            if task == "regression":
                st.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f}")
                import math
                rmse = math.sqrt(mean_squared_error(y_test, y_pred))
                st.metric("RMSE", f"{rmse:.2f}")

                st.metric("R²", f"{r2_score(y_test, y_pred):.2f}")
            else:
                st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
                st.text(classification_report(y_test, y_pred))

            st.subheader("🩺 Health Insights for Predicted Data")
            try:
                avg_pred, pred_kind = safe_avg_from_predictions(np.array(y_pred))
                cat, color = aqi_to_category(avg_pred)
                st.markdown(
                    f"<div style='background:{color};padding:10px;border-radius:8px;color:white'>"
                    f"🌡️ Predicted Mean AQI: {avg_pred:.0f} ({cat})</div>",
                    unsafe_allow_html=True,
                )
                for tip in HEALTH_ADVICE.get(cat, []):
                    st.write(f"- {tip}")
            except ValueError:
                st.warning("Model predictions could not be interpreted as AQI values or categories.")

            if "station" in df.columns:
                try:
                    df["predicted_aqi"] = pipe.predict(df[features])
                    if not pd.api.types.is_numeric_dtype(df["predicted_aqi"]):
                        df["predicted_aqi"] = df["predicted_aqi"].map(lambda x: CATEGORY_TO_MID.get(str(x).title(), np.nan))
                    by_station = df.groupby("station")["predicted_aqi"].mean().dropna()
                    if not by_station.empty:
                        st.subheader(f"🏭 Station-wise Predicted AQI — {city_choice}")
                        fig_pie = px.pie(names=by_station.index, values=by_station.values, hole=0.4)
                        st.plotly_chart(fig_pie, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not plot station-wise AQI: {e}")

            buf = io.BytesIO()
            joblib.dump(pipe, buf)
            st.download_button("⬇️ Download Trained Model", buf.getvalue(), file_name="aqi_model.joblib")
