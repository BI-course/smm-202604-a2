"""
Streamlit app (Streamlit Community Cloud) -- one tab per model, Group A2.

Like the Gradio app, this imports the SAME api.py the Flask/Render
deployment uses and calls its routes through Flask's test client, so every
platform serves predictions from the same trained artifacts and the same
code. Streamlit Community Cloud deploys straight from the GitHub repository,
so api.py and model/ are found one folder up from this file.
"""

import json
import os
import sys

import streamlit as st

# Find api.py: one level up (repository layout) or next to this file.
HERE = os.path.dirname(os.path.abspath(__file__))
for candidate in (os.path.dirname(HERE), HERE):
    if os.path.exists(os.path.join(candidate, "api.py")):
        sys.path.insert(0, candidate)
        os.chdir(candidate)  # so api.py's MODEL_DIR = "./model" resolves
        break


# Streamlit re-runs this whole script on every click. @st.cache_resource
# imports api.py (and therefore loads the four .joblib files) only once.
@st.cache_resource
def load_api():
    import api
    return api, api.app.test_client()


api, client = load_api()


def call_api(path, payload):
    # Send the JSON text ourselves: Flask's json= helper sorts the keys, and
    # a model whose scaler checks column order would then reject the request.
    response = client.post(path, data=json.dumps(payload), content_type="application/json")
    data = response.get_json(silent=True)
    if response.status_code != 200 or data is None:
        message = data.get("error") if isinstance(data, dict) else None
        st.error(message or f"The API returned HTTP {response.status_code}.")
        return None
    return data


def categories_for(pipeline, column):
    """Return the categories a fitted pipeline's encoder knows for a column."""
    try:
        for _, step in pipeline.steps:
            for _, transformer, columns in getattr(step, "transformers_", []):
                columns = list(columns) if not isinstance(columns, str) else [columns]
                if column not in columns:
                    continue
                inner_steps = getattr(transformer, "named_steps", {"t": transformer})
                for inner in inner_steps.values():
                    if hasattr(inner, "categories_"):
                        return [str(c) for c in inner.categories_[columns.index(column)]]
    except Exception:
        pass
    return None


def label_for(name, optional=()):
    label = name.replace("_", " ").capitalize()
    return label + " (optional)" if name in optional else label


def text_input(pipeline, name, default, key):
    choices = categories_for(pipeline, name)
    if choices:
        index = choices.index(default) if default in choices else 0
        return st.selectbox(label_for(name), choices, index=index, key=f"{key}_{name}")
    return st.text_input(label_for(name), default, key=f"{key}_{name}")


def number_input(name, default, key, optional=()):
    return st.number_input(label_for(name, optional), value=default,
                           placeholder="Leave blank if unknown", key=f"{key}_{name}")


def missing_fields(payload, optional=()):
    """Basic error handling: every non-optional field must be filled in."""
    return [
        label_for(name) for name, value in payload.items()
        if name not in optional and (value is None or (isinstance(value, str) and not value.strip()))
    ]


st.set_page_config(page_title="Group A2 Model Demos", page_icon="\U0001F4CA", layout="centered")
st.title("Group A2 -- Model Demos")

tab_regression, tab_classification, tab_clustering, tab_arm = st.tabs([
    "Property Rent Regressor", "Employee Attrition Classifier",
    "Retail Store Segmenter", "Music Recommender",
])

# -----------------------------------------------------------------------
# 1. Regression -- monthly rent (KES)
# -----------------------------------------------------------------------
with tab_regression:
    optional = ("garage_size_sqm", "security_rating")
    with st.form("regression_form"):
        payload = {
            "property_type": text_input(api.regression_pipeline, "property_type", "Apartment", "reg"),
            "neighborhood": text_input(api.regression_pipeline, "neighborhood", "Kilimani", "reg"),
            "furnishing_status": text_input(api.regression_pipeline, "furnishing_status", "Fully Furnished", "reg"),
            "property_age_years": number_input("property_age_years", 5.0, "reg"),
            "size_sqm": number_input("size_sqm", 100.0, "reg"),
            "num_bedrooms": number_input("num_bedrooms", 3, "reg"),
            "num_bathrooms": number_input("num_bathrooms", 2, "reg"),
            "distance_to_cbd_km": number_input("distance_to_cbd_km", 8.0, "reg"),
            "service_charge_kes": number_input("service_charge_kes", 5000, "reg"),
            "garage_size_sqm": number_input("garage_size_sqm", 15.0, "reg", optional),
            "security_rating": number_input("security_rating", 4.0, "reg", optional),
            "amenities_score": number_input("amenities_score", 80.0, "reg"),
            "nearby_schools_count": number_input("nearby_schools_count", 4, "reg"),
        }
        submitted = st.form_submit_button("Predict rent")
    if submitted:
        missing = missing_fields(payload, optional)
        if missing:
            st.error("Please fill in: " + ", ".join(missing))
        else:
            data = call_api("/api/v1/models/property-rent-regressor/predictions", payload)
            if data:
                st.success(f"Predicted monthly rent: KES {data['predicted_monthly_rent_kes']:,.2f}")
                st.caption(f"Model: {data['model']}")

# -----------------------------------------------------------------------
# 2. Classification -- employee attrition
# -----------------------------------------------------------------------
with tab_classification:
    optional = ("work_life_balance_score", "bonus_amount_kes")
    with st.form("classification_form"):
        payload = {
            "department": text_input(api.classification_pipeline, "department", "Operations", "clf"),
            "job_role_level": text_input(api.classification_pipeline, "job_role_level", "Junior", "clf"),
            "education_level": text_input(api.classification_pipeline, "education_level", "Diploma", "clf"),
            "performance_rating": text_input(api.classification_pipeline, "performance_rating", "Low", "clf"),
            "years_at_company": number_input("years_at_company", 1.5, "clf"),
            "years_since_last_promotion": number_input("years_since_last_promotion", 6.0, "clf"),
            "monthly_salary_kes": number_input("monthly_salary_kes", 25000, "clf"),
            "overtime_hours_per_month": number_input("overtime_hours_per_month", 45.0, "clf"),
            "job_satisfaction_score": number_input("job_satisfaction_score", 1.5, "clf"),
            "work_life_balance_score": number_input("work_life_balance_score", None, "clf", optional),
            "distance_from_home_km": number_input("distance_from_home_km", 35.0, "clf"),
            "num_companies_worked_previously": number_input("num_companies_worked_previously", 6, "clf"),
            "remote_work_days_per_week": number_input("remote_work_days_per_week", 0, "clf"),
            "bonus_amount_kes": number_input("bonus_amount_kes", None, "clf", optional),
        }
        submitted = st.form_submit_button("Predict attrition")
    if submitted:
        missing = missing_fields(payload, optional)
        if missing:
            st.error("Please fill in: " + ", ".join(missing))
        else:
            data = call_api("/api/v1/models/employee-attrition-classifier/predictions", payload)
            if data:
                st.success(f"Will the employee leave? {data['predicted_attrition']}")
                for label, prob in data.get("class_probabilities", {}).items():
                    st.write(f"{label}: {prob:.1%}")
                st.caption(f"Model: {data['model']}")

# -----------------------------------------------------------------------
# 3. Clustering -- retail store segment
# -----------------------------------------------------------------------
with tab_clustering:
    with st.form("clustering_form"):
        payload = {
            "monthly_revenue_kes": number_input("monthly_revenue_kes", 620000, "clu"),
            "daily_foot_traffic": number_input("daily_foot_traffic", 610, "clu"),
            "staff_count": number_input("staff_count", 7, "clu"),
            "store_size_sqm": number_input("store_size_sqm", 210, "clu"),
            "avg_basket_size_kes": number_input("avg_basket_size_kes", 55, "clu"),
        }
        submitted = st.form_submit_button("Assign segment")
    if submitted:
        missing = missing_fields(payload)
        if missing:
            st.error("Please fill in: " + ", ".join(missing))
        else:
            data = call_api("/api/v1/models/retail-store-segmenter/predictions", payload)
            if data:
                st.success(f"Assigned segment: Cluster {data['assigned_cluster']}")
                st.caption(f"Model: {data['model']}")

# -----------------------------------------------------------------------
# 4. Association rules -- music track recommendations
# -----------------------------------------------------------------------
with tab_arm:
    tracks = api.track_lookup.sort_index()
    with st.form("arm_form"):
        cart = st.multiselect(
            "Tracks in the listening session",
            options=list(tracks.index),
            default=["M09", "M10"],
            format_func=lambda t: f"{t} - {tracks.loc[t, 'item_name']} ({tracks.loc[t, 'category']})",
        )
        top_n = st.number_input("Top N recommendations", min_value=1, max_value=10, value=3)
        submitted = st.form_submit_button("Recommend")
    if submitted:
        if not cart:
            st.error("Pick at least one track for the listening session.")
        else:
            data = call_api("/api/v1/models/music-recommender/predictions",
                            {"cart": cart, "top_n": int(top_n)})
            if data:
                if not data["recommendations"]:
                    st.info("No recommendations for this session -- try e.g. M09 + M10, or M01 + M03.")
                for rec in data["recommendations"]:
                    st.write(f"**{rec['track_name']}** ({rec['track_id']}, {rec['category']}) -- "
                             f"confidence {rec['confidence']:.1%}, lift {rec['lift']:.2f}")
