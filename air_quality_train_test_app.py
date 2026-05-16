# ===============================================================
# 🌫️ Real-Time + Train/Test Air Quality ML System (Fully Fixed)
# ===============================================================

import os
import io
import requests
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
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# ===============================================================
# LOAD ENV VARIABLES
# ===============================================================

load_dotenv()

API_KEY = os.getenv("API_KEY")

# ✅ WORKING RESOURCE ID
RESOURCE_ID = os.getenv(
    "RESOURCE_ID",
    "c4f4c0f5-5f1f-4c8f-9f4f-d0f2f0b5b5d9"
)

# ===============================================================
# AQI BANDS
# ===============================================================

AQI_BANDS = [
    (0, 50, "Good", "#009865"),
    (51, 100, "Satisfactory", "#8BC34A"),
    (101, 200, "Moderate", "#FFC107"),
    (201, 300, "Poor", "#FF9800"),
    (301, 400, "Very Poor", "#F44336"),
    (401, 500, "Severe", "#7E0023"),
]

CATEGORY_TO_MID = {
    name: (lo + hi) / 2 for (lo, hi, name, _) in AQI_BANDS
}

HEALTH_ADVICE = {
    "Good": [
        "Air quality is satisfactory.",
        "Enjoy outdoor activities.",
    ],
    "Satisfactory": [
        "Minor breathing discomfort possible.",
        "Normal outdoor activities are safe.",
    ],
    "Moderate": [
        "Sensitive people should wear masks.",
        "Limit prolonged outdoor exposure.",
    ],
    "Poor": [
        "Avoid outdoor activities.",
        "Use air purifiers indoors.",
    ],
    "Very Poor": [
        "Stay indoors when possible.",
        "Avoid physical exertion outdoors.",
    ],
    "Severe": [
        "Hazardous air quality.",
        "Wear N95 masks and stay indoors.",
    ],
}

# ===============================================================
# FUNCTIONS
# ===============================================================

def aqi_to_category(aqi):
    try:
        aqi = float(aqi)
    except:
        return "Unknown", "#9E9E9E"

    for lo, hi, cat, color in AQI_BANDS:
        if lo <= aqi <= hi:
            return cat, color

    return "Severe", AQI_BANDS[-1][3]


def safe_avg_from_predictions(y_pred):
    try:
        return float(np.mean(y_pred.astype(float))), "numeric"

    except:
        try:
            mapped = [
                CATEGORY_TO_MID.get(str(x).title(), np.nan)
                for x in y_pred
            ]

            if np.isnan(mapped).all():
                raise ValueError

            return float(np.nanmean(mapped)), "category"

        except:
            raise ValueError(
                "Predictions are non-numeric and not AQI categories."
            )


# ===============================================================
# STREAMLIT PAGE
# ===============================================================

st.set_page_config(
    page_title="Real-Time AQI Dashboard",
    layout="wide"
)

st.title("🌫️ Real-Time Indian Air Quality + ML Prediction System")

mode = st.sidebar.radio(
    "Select Mode",
    [
        "📡 Real-Time AQI Dashboard",
        "🤖 Train/Test ML Model"
    ]
)

# ===============================================================
# REAL-TIME AQI DASHBOARD
# ===============================================================

if mode == "📡 Real-Time AQI Dashboard":

    st.header("🇮🇳 Real-Time Air Quality Monitoring")

    if not API_KEY:
        st.error("❌ API_KEY missing in .env file")

    else:

        # =======================================================
        # FETCH CITY LIST
        # =======================================================

        with st.spinner("Fetching city list..."):

            try:

                url_all = (
                    f"https://api.data.gov.in/resource/"
                    f"{RESOURCE_ID}"
                    f"?api-key={API_KEY}"
                    f"&format=json"
                    f"&limit=10000"
                )

                headers = {
                    "User-Agent": "Mozilla/5.0"
                }

                res = requests.get(
                    url_all,
                    headers=headers,
                    timeout=20
                )

                # DEBUG
                st.caption(f"API Status Code: {res.status_code}")

                if res.status_code != 200:
                    st.error(
                        f"❌ API Error: {res.status_code}"
                    )

                    st.text(res.text[:500])

                    cities = []

                else:

                    try:
                        data_all = res.json()

                    except Exception:
                        st.error("❌ Invalid JSON response")
                        st.text(res.text[:500])
                        cities = []

                    if (
                        "records" in data_all
                        and len(data_all["records"]) > 0
                    ):

                        df_all = pd.DataFrame(
                            data_all["records"]
                        )

                        if "city" in df_all.columns:

                            cities = sorted(
                                df_all["city"]
                                .dropna()
                                .astype(str)
                                .unique()
                                .tolist()
                            )

                        else:
                            st.error(
                                "❌ City column not found"
                            )
                            cities = []

                    else:
                        st.error("❌ No records found")
                        cities = []

            except requests.exceptions.Timeout:
                st.error("❌ API timeout")
                cities = []

            except requests.exceptions.ConnectionError:
                st.error("❌ Internet connection error")
                cities = []

            except Exception as e:
                st.error(f"❌ Error: {e}")
                cities = []

        # =======================================================
        # CITY SELECTION
        # =======================================================

        if not cities:

            st.warning(
                "⚠️ Could not load city list from API"
            )

        else:

            city = st.selectbox(
                "Select City",
                cities,
                index=cities.index("Delhi")
                if "Delhi" in cities
                else 0
            )

            # ===================================================
            # FETCH LIVE AQI DATA
            # ===================================================

            if st.button("Fetch Live AQI Data"):

                with st.spinner(
                    f"Fetching AQI data for {city}..."
                ):

                    try:

                        url = (
                            f"https://api.data.gov.in/resource/"
                            f"{RESOURCE_ID}"
                            f"?api-key={API_KEY}"
                            f"&format=json"
                            f"&filters[city]={city}"
                            f"&limit=500"
                        )

                        headers = {
                            "User-Agent": "Mozilla/5.0"
                        }

                        res = requests.get(
                            url,
                            headers=headers,
                            timeout=20
                        )

                        if res.status_code != 200:

                            st.error(
                                f"❌ API Error: {res.status_code}"
                            )

                        else:

                            try:
                                data = res.json()

                            except Exception:
                                st.error(
                                    "❌ Invalid JSON response"
                                )
                                st.text(res.text[:500])
                                st.stop()

                            if (
                                "records" not in data
                                or not data["records"]
                            ):

                                st.warning(
                                    "⚠️ No AQI records found"
                                )

                            else:

                                df = pd.DataFrame(
                                    data["records"]
                                )

                                st.subheader(
                                    f"📍 City: {city}"
                                )

                                st.dataframe(
                                    df.head(20),
                                    use_container_width=True
                                )

                                # ===================================
                                # AQI COLUMN DETECTION
                                # ===================================

                                possible_cols = [
                                    "pollutant_avg",
                                    "avg_value",
                                    "pollutant_avg_value",
                                    "pollutant_mean",
                                    "value",
                                    "aqi",
                                    "aqi_value",
                                ]

                                aqi_col = next(
                                    (
                                        c
                                        for c in possible_cols
                                        if c in df.columns
                                    ),
                                    None,
                                )

                                if not aqi_col:

                                    st.error(
                                        "❌ No AQI column found"
                                    )

                                else:

                                    st.success(
                                        f"Using AQI Column: {aqi_col}"
                                    )

                                    df[aqi_col] = pd.to_numeric(
                                        df[aqi_col],
                                        errors="coerce",
                                    )

                                    if (
                                        df[aqi_col]
                                        .notna()
                                        .sum()
                                        == 0
                                    ):

                                        st.warning(
                                            "⚠️ No numeric AQI values"
                                        )

                                    else:

                                        avg_aqi = (
                                            df[aqi_col].mean()
                                        )

                                        cat, color = (
                                            aqi_to_category(
                                                avg_aqi
                                            )
                                        )

                                        # ===========================
                                        # AQI DISPLAY
                                        # ===========================

                                        st.markdown(
                                            f"""
                                            <div style='
                                            background:{color};
                                            padding:15px;
                                            border-radius:10px;
                                            color:white;
                                            font-size:24px;
                                            '>

                                            🌡️ Average AQI in {city}:
                                            <b>{avg_aqi:.0f}</b>
                                            ({cat})

                                            </div>
                                            """,
                                            unsafe_allow_html=True,
                                        )

                                        # ===========================
                                        # HEALTH ADVICE
                                        # ===========================

                                        st.write(
                                            "## 🩺 Health Advice"
                                        )

                                        for tip in HEALTH_ADVICE.get(
                                            cat,
                                            [],
                                        ):
                                            st.write(f"- {tip}")

                                        # ===========================
                                        # HEALTH RISK
                                        # ===========================

                                        risk_score = min(
                                            max(
                                                (
                                                    avg_aqi
                                                    / 500
                                                )
                                                * 100,
                                                0,
                                            ),
                                            100,
                                        )

                                        if risk_score < 20:
                                            level = "Low"
                                            risk_color = "#00C853"

                                        elif risk_score < 40:
                                            level = "Mild"
                                            risk_color = "#AEEA00"

                                        elif risk_score < 60:
                                            level = "Moderate"
                                            risk_color = "#FFD600"

                                        elif risk_score < 80:
                                            level = "High"
                                            risk_color = "#FF6D00"

                                        else:
                                            level = "Critical"
                                            risk_color = "#D50000"

                                        st.markdown(
                                            f"""
                                            <div style='
                                            background:{risk_color};
                                            padding:12px;
                                            border-radius:10px;
                                            color:white;
                                            font-size:20px;
                                            '>

                                            💀 Health Risk:
                                            <b>{level}</b>

                                            </div>
                                            """,
                                            unsafe_allow_html=True,
                                        )

                                        st.progress(
                                            int(risk_score)
                                        )

                                        # ===========================
                                        # PIE CHART
                                        # ===========================

                                        if "station" in df.columns:

                                            st.write(
                                                "## 🏭 Station-wise AQI Distribution"
                                            )

                                            station_avg = (
                                                df.groupby(
                                                    "station"
                                                )[aqi_col]
                                                .mean()
                                                .dropna()
                                            )

                                            if not station_avg.empty:

                                                fig = px.pie(
                                                    names=station_avg.index,
                                                    values=station_avg.values,
                                                    title=f"AQI Share by Station - {city}",
                                                    hole=0.4,
                                                )

                                                st.plotly_chart(
                                                    fig,
                                                    use_container_width=True,
                                                )

                    except Exception as e:
                        st.error(f"❌ Error: {e}")

# ===============================================================
# TRAIN / TEST MODEL
# ===============================================================

else:

    st.header("🤖 Train & Evaluate AQI ML Model")

    file = st.sidebar.file_uploader(
        "Upload Dataset (CSV)",
        type=["csv"]
    )

    if not file:

        st.info(
            "Upload a CSV dataset to train ML model"
        )

    else:

        df = pd.read_csv(file)

        st.dataframe(
            df.head(),
            use_container_width=True
        )

        # =======================================================
        # TARGET COLUMN
        # =======================================================

        possible_targets = [
            c
            for c in df.columns
            if "aqi" in c.lower()
            or "pollutant_avg" in c.lower()
        ]

        target_default = (
            possible_targets[0]
            if possible_targets
            else df.columns[-1]
        )

        target = st.sidebar.selectbox(
            "🎯 Target Column",
            df.columns,
            index=list(df.columns).index(
                target_default
            ),
        )

        # =======================================================
        # CITY FILTER
        # =======================================================

        if "city" in df.columns:

            city_choice = st.selectbox(
                "Select City",
                ["All Cities"]
                + sorted(
                    df["city"]
                    .dropna()
                    .unique()
                    .tolist()
                ),
            )

            train_city_only = st.checkbox(
                "Train on selected city only",
                value=False,
            )

            if (
                train_city_only
                and city_choice != "All Cities"
            ):

                df = df[df["city"] == city_choice]

                st.success(
                    f"Training on {city_choice}"
                )

        # =======================================================
        # DATETIME FEATURES
        # =======================================================

        datetime_cols = [
            c
            for c in df.columns
            if "date" in c.lower()
            or "time" in c.lower()
        ]

        for col in datetime_cols:

            try:

                df[col] = pd.to_datetime(
                    df[col],
                    errors="coerce"
                )

                df[f"{col}_hour"] = (
                    df[col].dt.hour
                )

                df[f"{col}_month"] = (
                    df[col].dt.month
                )

            except:
                pass

        exclude_cols = [target] + datetime_cols

        features = st.sidebar.multiselect(
            "🧩 Feature Columns",
            [
                c
                for c in df.columns
                if c not in exclude_cols
            ],
            default=[
                c
                for c in df.columns
                if c not in exclude_cols
            ][:6],
        )

        test_size = st.sidebar.slider(
            "Test Size",
            0.1,
            0.4,
            0.2,
        )

        # =======================================================
        # TRAIN MODEL
        # =======================================================

        if st.button("🚀 Train Model"):

            df = df.dropna(subset=[target])

            X = df[features]
            y = df[target]

            task = (
                "regression"
                if pd.api.types.is_numeric_dtype(y)
                else "classification"
            )

            num_cols = [
                c
                for c in X.columns
                if pd.api.types.is_numeric_dtype(X[c])
            ]

            cat_cols = [
                c
                for c in X.columns
                if c not in num_cols
            ]

            pre = ColumnTransformer([
                (
                    "num",
                    Pipeline([
                        (
                            "imp",
                            SimpleImputer(
                                strategy="median"
                            ),
                        ),
                        (
                            "scaler",
                            StandardScaler(),
                        ),
                    ]),
                    num_cols,
                ),
                (
                    "cat",
                    Pipeline([
                        (
                            "imp",
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
                        ),
                        (
                            "ohe",
                            OneHotEncoder(
                                handle_unknown="ignore"
                            ),
                        ),
                    ]),
                    cat_cols,
                ),
            ])

            model = (
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=42,
                )
                if task == "regression"
                else RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                )
            )

            pipe = Pipeline([
                ("pre", pre),
                ("model", model),
            ])

            X_train, X_test, y_train, y_test = (
                train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=42,
                )
            )

            pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)

            st.success(
                "✅ Model trained successfully"
            )

            # ===================================================
            # EVALUATION
            # ===================================================

            st.subheader("📊 Model Evaluation")

            if task == "regression":

                st.metric(
                    "MAE",
                    f"{mean_absolute_error(y_test, y_pred):.2f}",
                )

                rmse = np.sqrt(
                    mean_squared_error(
                        y_test,
                        y_pred,
                    )
                )

                st.metric(
                    "RMSE",
                    f"{rmse:.2f}",
                )

                st.metric(
                    "R² Score",
                    f"{r2_score(y_test, y_pred):.2f}",
                )

            else:

                st.metric(
                    "Accuracy",
                    f"{accuracy_score(y_test, y_pred):.3f}",
                )

                st.text(
                    classification_report(
                        y_test,
                        y_pred,
                    )
                )

            # ===================================================
            # DOWNLOAD MODEL
            # ===================================================

            buf = io.BytesIO()

            joblib.dump(pipe, buf)

            st.download_button(
                "⬇️ Download Trained Model",
                buf.getvalue(),
                file_name="aqi_model.joblib",
            )