from flask import Flask, render_template, request, redirect, url_for, flash
import os
import uuid
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.secret_key = "hospital-resource-optimizer-secret-key"

UPLOAD_FOLDER = "uploads"
PLOT_FOLDER = os.path.join("static", "plots")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

df = None
filename = None


# =========================================================
# HELPERS
# =========================================================
def clear_plot_folder():
    for file in os.listdir(PLOT_FOLDER):
        path = os.path.join(PLOT_FOLDER, file)
        if os.path.isfile(path):
            try:
                os.remove(path)
            except Exception:
                pass


def save_plot(fig, prefix):
    name = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
    path = os.path.join(PLOT_FOLDER, name)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=140)
    plt.close(fig)
    return name


def normalize_name(name):
    return str(name).strip().lower().replace(" ", "_")


def build_column_lookup(columns):
    return {normalize_name(col): col for col in columns}


def find_column(dataframe, aliases):
    lookup = build_column_lookup(dataframe.columns)
    for alias in aliases:
        key = normalize_name(alias)
        if key in lookup:
            return lookup[key]
    return None


def try_parse_dates(dataframe):
    df_copy = dataframe.copy()
    date_col = find_column(df_copy, ["date", "record_date", "admission_date"])
    if date_col:
        parsed = pd.to_datetime(df_copy[date_col], errors="coerce")
        if parsed.notna().sum() > 0:
            df_copy[date_col] = parsed
    return df_copy, date_col


def get_season_from_month(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Summer"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    return "Autumn"


def safe_numeric(series, default=0):
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    if s.isnull().all():
        return pd.Series([default] * len(series), index=series.index)
    median_val = s.median()
    if pd.isna(median_val):
        median_val = default
    return s.fillna(median_val)


# =========================================================
# DATA ENRICHMENT
# =========================================================
def enrich_hospital_data(dataframe):
    data = dataframe.copy()
    data, date_col = try_parse_dates(data)

    patient_col = find_column(data, [
        "patient_load", "patients", "patient_count", "daily_patients",
        "admissions", "total_patients"
    ])
    bed_col = find_column(data, [
        "bed_occupancy", "occupied_beds", "beds_used", "bed_usage"
    ])
    los_col = find_column(data, [
        "average_length_of_stay", "length_of_stay", "avg_stay",
        "average_stay", "los"
    ])
    severity_col = find_column(data, [
        "severity_score", "severity", "acuity_score"
    ])
    emergency_col = find_column(data, [
        "emergency_admissions", "emergency_cases", "emergency_patients"
    ])
    available_beds_col = find_column(data, [
        "available_beds", "beds_available", "total_beds"
    ])
    staff_col = find_column(data, [
        "staff_count", "staff", "total_staff"
    ])
    icu_col = find_column(data, [
        "icu_beds", "icu_capacity", "available_icu_beds"
    ])
    season_col = find_column(data, ["season"])
    weather_col = find_column(data, ["weather_condition", "weather"])

    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()

    if patient_col is None and numeric_cols:
        patient_col = numeric_cols[0]

    if patient_col is None:
        raise ValueError("No usable numeric column found for prediction.")

    patient_series = safe_numeric(data[patient_col], default=0)
    data[patient_col] = patient_series

    if date_col:
        data[f"{date_col}_year"] = data[date_col].dt.year
        data[f"{date_col}_month"] = data[date_col].dt.month
        data[f"{date_col}_day"] = data[date_col].dt.day
        data[f"{date_col}_weekday_num"] = data[date_col].dt.weekday
        data["Day_of_Week_Derived"] = data[date_col].dt.day_name()

        if season_col is None:
            data["Season_Derived"] = data[date_col].dt.month.apply(
                lambda x: get_season_from_month(x) if not pd.isna(x) else "General"
            )
            season_col = "Season_Derived"

    if emergency_col and emergency_col in data.columns:
        emergency_vals = safe_numeric(data[emergency_col], default=0)
        data[emergency_col] = emergency_vals
        denom = patient_series.replace(0, np.nan)
        data["Emergency_Ratio"] = (emergency_vals / denom).replace([np.inf, -np.inf], np.nan).fillna(0)

    if bed_col is None:
        if available_beds_col and available_beds_col in data.columns:
            available_vals = safe_numeric(data[available_beds_col], default=150).replace(0, np.nan)
            data[available_beds_col] = available_vals.fillna(150)
            bed_occ = (patient_series / available_vals) * 100
            data["Bed_Occupancy_Derived"] = bed_occ.replace([np.inf, -np.inf], np.nan).fillna(0)
            bed_col = "Bed_Occupancy_Derived"
        else:
            data["Bed_Occupancy_Derived"] = (patient_series * 0.75).fillna(0)
            bed_col = "Bed_Occupancy_Derived"
    else:
        data[bed_col] = safe_numeric(data[bed_col], default=0)

    if los_col is None:
        max_patient = patient_series.max() if patient_series.max() > 0 else 1
        data["Average_Length_of_Stay_Derived"] = np.clip(2 + (patient_series / max_patient) * 5, 2, 9)
        los_col = "Average_Length_of_Stay_Derived"
    else:
        data[los_col] = safe_numeric(data[los_col], default=0)

    if "High_Load_Flag" not in data.columns:
        threshold = patient_series.quantile(0.75)
        data["High_Load_Flag"] = (patient_series >= threshold).astype(int)
    else:
        data["High_Load_Flag"] = safe_numeric(data["High_Load_Flag"], default=0).astype(int)

    if "Inventory_Demand" not in data.columns:
        data["Inventory_Demand"] = np.ceil(patient_series * 1.8)
    else:
        data["Inventory_Demand"] = safe_numeric(data["Inventory_Demand"], default=0)

    if severity_col and severity_col in data.columns:
        data[severity_col] = safe_numeric(data[severity_col], default=0)

    if available_beds_col and available_beds_col in data.columns:
        data[available_beds_col] = safe_numeric(data[available_beds_col], default=150)

    if staff_col and staff_col in data.columns:
        data[staff_col] = safe_numeric(data[staff_col], default=60)

    if icu_col and icu_col in data.columns:
        data[icu_col] = safe_numeric(data[icu_col], default=20)

    if season_col is None:
        data["Season_Derived"] = "General"
        season_col = "Season_Derived"

    if weather_col is None:
        data["Weather_Derived"] = "Normal"
        weather_col = "Weather_Derived"

    data = data.replace([np.inf, -np.inf], np.nan).fillna(0)

    return data, {
        "date_col": date_col,
        "patient_col": patient_col,
        "bed_col": bed_col,
        "los_col": los_col,
        "severity_col": severity_col,
        "emergency_col": emergency_col,
        "available_beds_col": available_beds_col,
        "staff_col": staff_col,
        "icu_col": icu_col,
        "season_col": season_col,
        "weather_col": weather_col,
        "high_load_col": "High_Load_Flag",
        "inventory_col": "Inventory_Demand",
    }


# =========================================================
# PREPROCESSING
# =========================================================
def preprocess_for_models(dataframe):
    processed = dataframe.copy()
    steps = []

    duplicate_before = int(processed.duplicated().sum())
    if duplicate_before > 0:
        processed = processed.drop_duplicates()
        steps.append(f"Removed {duplicate_before} duplicate rows")
    else:
        steps.append("Checked duplicate rows")

    missing_before = int(processed.isnull().sum().sum())
    if missing_before > 0:
        steps.append("Handled missing values")
    else:
        steps.append("No missing values found")

    for col in list(processed.columns):
        if pd.api.types.is_datetime64_any_dtype(processed[col]):
            processed[f"{col}_year"] = processed[col].dt.year
            processed[f"{col}_month"] = processed[col].dt.month
            processed[f"{col}_day"] = processed[col].dt.day
            processed[f"{col}_weekday"] = processed[col].dt.weekday
            processed.drop(columns=[col], inplace=True)

    encoded_columns = []

    for col in processed.columns:
        if processed[col].dtype == "object":
            processed[col] = processed[col].fillna("Unknown").astype(str).replace("", "Unknown")
            le = LabelEncoder()
            processed[col] = le.fit_transform(processed[col])
            encoded_columns.append(col)
        else:
            processed[col] = pd.to_numeric(processed[col], errors="coerce")

    if encoded_columns:
        steps.append("Encoded categorical columns")
    else:
        steps.append("No categorical encoding needed")

    numeric_cols = processed.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        processed[col] = processed[col].replace([np.inf, -np.inf], np.nan)
        if processed[col].isnull().sum() > 0:
            median_val = processed[col].median()
            if pd.isna(median_val):
                median_val = 0
            processed[col] = processed[col].fillna(median_val)

    processed = processed.replace([np.inf, -np.inf], np.nan).fillna(0)

    # IMPORTANT: NO SCALING HERE
    steps.append("Kept numeric values in original scale for realistic predictions")

    return processed, steps


# =========================================================
# TODAY INPUT
# =========================================================
def prepare_today_input(original_df, column_map, form_data=None):
    base_row = original_df.tail(1).copy()

    if base_row.empty:
        raise ValueError("Dataset is empty.")

    if form_data:
        def assign_if_exists(col_aliases, value, cast_type="float"):
            col = find_column(base_row, col_aliases)
            if col and value not in [None, "", "None"]:
                try:
                    if cast_type == "float":
                        base_row.loc[base_row.index[-1], col] = float(value)
                    else:
                        base_row.loc[base_row.index[-1], col] = str(value)
                except Exception:
                    pass

        assign_if_exists(
            ["patient_load", "patients", "patient_count", "daily_patients", "admissions"],
            form_data.get("patient_load"),
            "float"
        )
        assign_if_exists(
            ["emergency_admissions", "emergency_cases", "emergency_patients"],
            form_data.get("emergency_admissions"),
            "float"
        )
        assign_if_exists(
            ["severity_score", "severity", "acuity_score"],
            form_data.get("severity_score"),
            "float"
        )
        assign_if_exists(
            ["available_beds", "beds_available", "total_beds"],
            form_data.get("available_beds"),
            "float"
        )
        assign_if_exists(
            ["staff_count", "staff", "total_staff"],
            form_data.get("staff_count"),
            "float"
        )
        assign_if_exists(
            ["icu_beds", "icu_capacity", "available_icu_beds"],
            form_data.get("icu_beds"),
            "float"
        )
        assign_if_exists(["season"], form_data.get("season"), "text")
        assign_if_exists(["weather_condition", "weather"], form_data.get("weather_condition"), "text")
        assign_if_exists(["day_of_week"], form_data.get("day_of_week"), "text")

        date_col = column_map.get("date_col")
        if date_col and form_data.get("date"):
            try:
                base_row.loc[base_row.index[-1], date_col] = pd.to_datetime(form_data.get("date"))
            except Exception:
                pass

    return base_row


# =========================================================
# EDA
# =========================================================
def generate_eda_summary(dataframe, column_map):
    insights = []

    rows, cols = dataframe.shape
    insights.append(f"Dataset contains {rows} rows and {cols} columns.")
    insights.append(f"Missing values: {int(dataframe.isnull().sum().sum())}")
    insights.append(f"Duplicate rows: {int(dataframe.duplicated().sum())}")

    patient_col = column_map["patient_col"]
    bed_col = column_map["bed_col"]
    severity_col = column_map["severity_col"]

    if patient_col in dataframe.columns:
        patient_series = safe_numeric(dataframe[patient_col], default=0)
        insights.append(
            f"Average patient load is {round(patient_series.mean(), 2)} and peak load is {round(patient_series.max(), 2)}."
        )

    if bed_col in dataframe.columns:
        bed_series = safe_numeric(dataframe[bed_col], default=0)
        insights.append(f"Average bed occupancy is {round(bed_series.mean(), 2)}.")

    if severity_col and severity_col in dataframe.columns:
        sev = safe_numeric(dataframe[severity_col], default=0)
        pat = safe_numeric(dataframe[patient_col], default=0)
        if len(sev) > 1 and len(pat) > 1:
            corr = sev.corr(pat)
            if pd.notna(corr):
                insights.append(f"Severity score correlation with patient load: {round(corr, 2)}.")

    if column_map["high_load_col"] in dataframe.columns:
        high_days = int(safe_numeric(dataframe[column_map["high_load_col"]], default=0).sum())
        insights.append(f"High load days detected: {high_days}")

    return insights


# =========================================================
# PLOTS
# =========================================================
def generate_plots(dataframe, column_map):
    plots = {}
    date_col = column_map["date_col"]
    patient_col = column_map["patient_col"]
    bed_col = column_map["bed_col"]
    severity_col = column_map["severity_col"]
    season_col = column_map["season_col"]

    if patient_col in dataframe.columns:
        fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="#0a1020")
        ax.set_facecolor("#121a2f")
        y = safe_numeric(dataframe[patient_col], default=0)

        if date_col and date_col in dataframe.columns and pd.api.types.is_datetime64_any_dtype(dataframe[date_col]):
            ax.plot(dataframe[date_col], y, linewidth=2.5)
            ax.set_xlabel("Date", color="white")
        else:
            ax.plot(range(len(dataframe)), y, linewidth=2.5)
            ax.set_xlabel("Record Index", color="white")

        ax.set_ylabel("Patient Load", color="white")
        ax.set_title("Patient Load Trend", color="white", fontsize=14)
        ax.tick_params(colors="white")
        plots["patient_trend"] = save_plot(fig, "patient_trend")

    if bed_col in dataframe.columns:
        fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="#0a1020")
        ax.set_facecolor("#121a2f")
        bed_series = safe_numeric(dataframe[bed_col], default=0)

        if date_col and date_col in dataframe.columns and pd.api.types.is_datetime64_any_dtype(dataframe[date_col]):
            temp = pd.DataFrame({
                "month": dataframe[date_col].dt.month_name(),
                "bed": bed_series
            })
            month_order = ["January", "February", "March", "April", "May", "June",
                           "July", "August", "September", "October", "November", "December"]
            grouped = temp.groupby("month")["bed"].mean().reindex(month_order).dropna()
            ax.bar(grouped.index, grouped.values)
            ax.set_xticks(range(len(grouped.index)))
            ax.set_xticklabels(grouped.index, rotation=45, ha="right", color="white")
            ax.set_title("Average Bed Occupancy by Month", color="white")
        else:
            grouped = bed_series.groupby(np.arange(len(bed_series)) % 7).mean()
            ax.bar(grouped.index.astype(str), grouped.values)
            ax.set_title("Average Bed Occupancy by Record Pattern", color="white")

        ax.tick_params(colors="white")
        plots["bed_month"] = save_plot(fig, "bed_month")

    if season_col and season_col in dataframe.columns and patient_col in dataframe.columns:
        fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="#0a1020")
        ax.set_facecolor("#121a2f")

        temp = pd.DataFrame({
            "season": dataframe[season_col].astype(str),
            "patient": safe_numeric(dataframe[patient_col], default=0)
        })
        grouped = temp.groupby("season")["patient"].mean().sort_values(ascending=False)
        ax.bar(grouped.index, grouped.values)
        ax.set_title("Season-wise Patient Load Pattern", color="white")
        ax.tick_params(colors="white")
        plots["season_pattern"] = save_plot(fig, "season_pattern")

    if severity_col and severity_col in dataframe.columns and patient_col in dataframe.columns:
        fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="#0a1020")
        ax.set_facecolor("#121a2f")

        x = safe_numeric(dataframe[severity_col], default=0)
        y = safe_numeric(dataframe[patient_col], default=0)
        ax.scatter(x, y, alpha=0.7)
        ax.set_xlabel("Severity Score", color="white")
        ax.set_ylabel("Patient Load", color="white")
        ax.set_title("Severity Score vs Patient Load", color="white")
        ax.tick_params(colors="white")
        plots["severity_scatter"] = save_plot(fig, "severity_scatter")

    numeric_cols = dataframe.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) >= 2:
        numeric_df = dataframe[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0a1020")
        ax.set_facecolor("#121a2f")
        corr = numeric_df.corr(numeric_only=True)
        sns.heatmap(corr, cmap="Blues", annot=True, fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap", color="white")
        plots["heatmap"] = save_plot(fig, "heatmap")

    return plots


# =========================================================
# MODELS
# =========================================================
def train_linear_model(processed_df, target_col, prediction_row_df):
    if target_col not in processed_df.columns:
        return None

    X = processed_df.drop(columns=[target_col]).copy()
    y = processed_df[target_col].copy()

    if X.empty or len(X) < 8:
        return None

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = pd.to_numeric(y, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)

    pred_X = prediction_row_df[X.columns].copy()
    pred_X = pred_X.replace([np.inf, -np.inf], np.nan).fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    future_pred = model.predict(pred_X)[0]

    # IMPORTANT FIX: no negative output
    future_pred = max(0, float(future_pred))

    return {
        "prediction": round(future_pred, 2),
        "mse": round(float(mean_squared_error(y_test, preds)), 4),
        "r2": round(float(r2_score(y_test, preds)), 4)
    }


def train_logistic_model(processed_df, target_col, prediction_row_df):
    if target_col not in processed_df.columns:
        return None

    X = processed_df.drop(columns=[target_col]).copy()
    y = processed_df[target_col].copy()

    if X.empty or len(X) < 8:
        return None

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = pd.to_numeric(y, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).astype(int)

    if pd.Series(y).nunique() < 2:
        return None

    pred_X = prediction_row_df[X.columns].copy()
    pred_X = pred_X.replace([np.inf, -np.inf], np.nan).fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    pred_class = int(model.predict(pred_X)[0])
    pred_prob = float(model.predict_proba(pred_X)[0][1])
    test_preds = model.predict(X_test)
    acc = accuracy_score(y_test, test_preds)

    return {
        "flag": "Yes" if pred_class == 1 else "No",
        "probability": round(pred_prob * 100, 2),
        "accuracy": round(float(acc), 4)
    }


# =========================================================
# RESOURCE RECOMMENDATION
# =========================================================
def build_resource_recommendation(original_df, predictions, column_map):
    available_beds_col = column_map.get("available_beds_col")
    staff_col = column_map.get("staff_col")
    icu_col = column_map.get("icu_col")

    latest_available_beds = 150
    latest_staff = 60
    latest_icu = 20

    if available_beds_col and available_beds_col in original_df.columns:
        latest_available_beds = int(safe_numeric(original_df[available_beds_col], default=150).iloc[-1])

    if staff_col and staff_col in original_df.columns:
        latest_staff = int(safe_numeric(original_df[staff_col], default=60).iloc[-1])

    if icu_col and icu_col in original_df.columns:
        latest_icu = int(safe_numeric(original_df[icu_col], default=20).iloc[-1])

    patient_load = max(0, float(predictions.get("patient_load_prediction", 0)))
    bed_occ = max(0, float(predictions.get("bed_occupancy_prediction", 0)))
    inventory = max(0, float(predictions.get("inventory_prediction", 0)))

    prepare_beds = int(np.ceil(patient_load * 0.82))
    extra_safety_beds = int(np.ceil(prepare_beds * 0.08))
    total_beds_to_prepare = prepare_beds + extra_safety_beds

    total_staff = int(np.ceil(patient_load / 3)) if patient_load > 0 else 0
    nurses = int(np.ceil(total_staff * 0.72))
    doctors = int(np.ceil(total_staff * 0.28))

    extra_shifts = 0
    if total_staff > latest_staff:
        extra_shifts = int(np.ceil((total_staff - latest_staff) / 8))

    inventory_level = "High" if inventory > patient_load * 1.6 else "Moderate"
    medicines = int(np.ceil(inventory * 0.50))
    surgical_items = int(np.ceil(inventory * 0.30))
    ppe = int(np.ceil(inventory * 0.20))
    icu_required = int(np.ceil(patient_load * 0.12))

    suggestions = [
        f"Prepare {total_beds_to_prepare} beds including {extra_safety_beds} extra safety beds.",
        f"Schedule {total_staff} total staff: {nurses} nurses and {doctors} doctors.",
        f"Add {extra_shifts} extra shifts for smooth operations." if extra_shifts > 0 else "Current shift structure is sufficient.",
        f"Estimated inventory demand is {inventory_level}. Keep extra medicines, PPE, and surgical items ready."
    ]

    return {
        "available_beds": latest_available_beds,
        "predicted_bed_occupancy": round(bed_occ, 2),
        "beds_to_prepare": total_beds_to_prepare,
        "available_staff": latest_staff,
        "recommended_staff": total_staff,
        "nurses_needed": nurses,
        "doctors_needed": doctors,
        "extra_shifts": extra_shifts,
        "available_icu": latest_icu,
        "recommended_icu": icu_required,
        "inventory_level": inventory_level,
        "medicines_required": medicines,
        "surgical_items_required": surgical_items,
        "ppe_required": ppe,
        "suggestions": suggestions
    }


# =========================================================
# PIPELINE
# =========================================================
def build_prediction_row(full_original_df, user_form=None):
    enriched_df, column_map = enrich_hospital_data(full_original_df)

    prediction_source = prepare_today_input(enriched_df, column_map, user_form)
    enriched_prediction_source, _ = enrich_hospital_data(prediction_source)

    for col in enriched_df.columns:
        if col not in enriched_prediction_source.columns:
            enriched_prediction_source[col] = np.nan

    for col in enriched_prediction_source.columns:
        if col not in enriched_df.columns:
            enriched_df[col] = np.nan

    enriched_prediction_source = enriched_prediction_source[enriched_df.columns]

    combined = pd.concat([enriched_df, enriched_prediction_source], ignore_index=True)
    processed_combined, preprocessing_steps = preprocess_for_models(combined)

    processed_combined = processed_combined.replace([np.inf, -np.inf], np.nan).fillna(0)

    processed_training = processed_combined.iloc[:-1].copy()
    processed_prediction_row = processed_combined.iloc[[-1]].copy()

    return enriched_df, processed_training, processed_prediction_row, column_map, preprocessing_steps


def run_full_pipeline(original_df, user_form=None):
    enriched_df, processed_training, processed_prediction_row, column_map, preprocessing_steps = build_prediction_row(
        original_df, user_form
    )

    patient_result = train_linear_model(processed_training, column_map["patient_col"], processed_prediction_row)
    bed_result = train_linear_model(processed_training, column_map["bed_col"], processed_prediction_row)
    los_result = train_linear_model(processed_training, column_map["los_col"], processed_prediction_row)
    inventory_result = train_linear_model(processed_training, column_map["inventory_col"], processed_prediction_row)
    logistic_result = train_logistic_model(processed_training, column_map["high_load_col"], processed_prediction_row)

    predictions = {
        "patient_load_prediction": patient_result["prediction"] if patient_result else 0,
        "bed_occupancy_prediction": bed_result["prediction"] if bed_result else 0,
        "average_los_prediction": los_result["prediction"] if los_result else 0,
        "inventory_prediction": inventory_result["prediction"] if inventory_result else 0,
        "high_load_flag": logistic_result["flag"] if logistic_result else "No",
        "high_load_probability": logistic_result["probability"] if logistic_result else 0,
        "patient_metrics": patient_result,
        "bed_metrics": bed_result,
        "los_metrics": los_result,
        "inventory_metrics": inventory_result,
        "logistic_metrics": logistic_result
    }

    resource_recommendation = build_resource_recommendation(enriched_df, predictions, column_map)
    eda_summary = generate_eda_summary(enriched_df, column_map)
    plots = generate_plots(enriched_df, column_map)

    return {
        "enriched_df": enriched_df,
        "column_map": column_map,
        "preprocessing_steps": preprocessing_steps,
        "predictions": predictions,
        "resource_recommendation": resource_recommendation,
        "eda_summary": eda_summary,
        "plots": plots
    }


def get_model_table():
    return [
        {
            "model_type": "Linear Regression",
            "target": "Patient Load",
            "purpose": "Predict tomorrow's total patient count",
            "output": "Continuous value"
        },
        {
            "model_type": "Linear Regression",
            "target": "Bed Occupancy",
            "purpose": "Estimate beds likely to be occupied",
            "output": "Continuous value"
        },
        {
            "model_type": "Linear Regression",
            "target": "Average Length of Stay",
            "purpose": "Support discharge and bed planning",
            "output": "Continuous value"
        },
        {
            "model_type": "Linear Regression",
            "target": "Inventory Demand",
            "purpose": "Estimate medicine and PPE requirement",
            "output": "Continuous value"
        },
        {
            "model_type": "Logistic Regression",
            "target": "High Load Flag",
            "purpose": "Predict whether tomorrow is a high load day",
            "output": "Yes / No + Probability"
        }
    ]


# =========================================================
# ROUTES
# =========================================================
@app.route("/")
def home():
    global df, filename

    if df is None:
        return render_template(
            "index.html",
            file_uploaded=False
        )

    try:
        clear_plot_folder()
        result = run_full_pipeline(df)

        return render_template(
            "index.html",
            file_uploaded=True,
            filename=filename,
            eda_summary=result["eda_summary"],
            preprocessing_steps=result["preprocessing_steps"],
            predictions=result["predictions"],
            resource_recommendation=result["resource_recommendation"],
            plots=result["plots"],
            columns=result["enriched_df"].columns.tolist(),
            table_preview=result["enriched_df"].head(10).to_dict(orient="records"),
            model_table=get_model_table(),
            detected_targets=result["column_map"],
            form_values={}
        )
    except Exception as e:
        flash(f"Error while processing dataset: {str(e)}", "error")
        return render_template("index.html", file_uploaded=False)


@app.route("/upload", methods=["POST"])
def upload():
    global df, filename

    if "file" not in request.files:
        flash("Please choose a CSV file.", "error")
        return redirect(url_for("home"))

    file = request.files["file"]

    if file.filename == "":
        flash("No file selected.", "error")
        return redirect(url_for("home"))

    try:
        if not file.filename.lower().endswith(".csv"):
            flash("Only CSV files are supported.", "error")
            return redirect(url_for("home"))

        saved_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(saved_path)

        df = pd.read_csv(saved_path)
        filename = file.filename
        clear_plot_folder()

        flash("Dataset uploaded successfully.", "success")
    except Exception as e:
        df = None
        filename = None
        flash(f"Upload failed: {str(e)}", "error")

    return redirect(url_for("home"))


@app.route("/predict_today", methods=["POST"])
def predict_today():
    global df, filename

    if df is None:
        flash("Upload a hospital CSV first.", "error")
        return redirect(url_for("home"))

    form_values = {
        "date": request.form.get("date", ""),
        "day_of_week": request.form.get("day_of_week", ""),
        "emergency_admissions": request.form.get("emergency_admissions", ""),
        "severity_score": request.form.get("severity_score", ""),
        "available_beds": request.form.get("available_beds", ""),
        "staff_count": request.form.get("staff_count", ""),
        "icu_beds": request.form.get("icu_beds", ""),
        "patient_load": request.form.get("patient_load", ""),
        "season": request.form.get("season", ""),
        "weather_condition": request.form.get("weather_condition", ""),
    }

    try:
        clear_plot_folder()
        result = run_full_pipeline(df, user_form=form_values)

        flash("Prediction generated using today's input.", "success")

        return render_template(
            "index.html",
            file_uploaded=True,
            filename=filename,
            eda_summary=result["eda_summary"],
            preprocessing_steps=result["preprocessing_steps"],
            predictions=result["predictions"],
            resource_recommendation=result["resource_recommendation"],
            plots=result["plots"],
            columns=result["enriched_df"].columns.tolist(),
            table_preview=result["enriched_df"].head(10).to_dict(orient="records"),
            model_table=get_model_table(),
            detected_targets=result["column_map"],
            form_values=form_values
        )
    except Exception as e:
        flash(f"Prediction failed: {str(e)}", "error")
        return redirect(url_for("home"))


@app.route("/reset")
def reset():
    global df, filename
    df = None
    filename = None
    clear_plot_folder()
    flash("Dashboard reset successfully.", "success")
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True)