"""
Gradio app (Hugging Face Space) -- one tab per model, Group A2.

Instead of loading the four .joblib files a second time and copying each
route's logic, this app imports the SAME api.py the Flask/Render deployment
uses and calls its routes through Flask's built-in test client. That
guarantees every platform (local, Docker/Nginx, Render, Hugging Face,
Streamlit) serves predictions from the same trained artifacts and the same
code -- nothing is retrained or re-implemented here.

Files the Space needs (all in the Space's root folder):
    app.py              <- this file
    api.py              <- copied unchanged from the repository root
    model/              <- the four .joblib files, copied unchanged
    requirements.txt
"""

import json
import os
import sys

# Find api.py: next to this file (inside the Space) or one level up (when run
# from this folder inside the repository). Change into that folder so
# api.py's MODEL_DIR = "./model" resolves correctly.
HERE = os.path.dirname(os.path.abspath(__file__))
for candidate in (HERE, os.path.dirname(HERE)):
    if os.path.exists(os.path.join(candidate, "api.py")):
        sys.path.insert(0, candidate)
        os.chdir(candidate)
        break

import gradio as gr

try:
    # Free-tier Gradio Spaces on ZeroGPU hardware expect at least one
    # @spaces.GPU function. None of these models needs a GPU; this only
    # satisfies that platform check. On CPU hardware it is skipped.
    import spaces

    @spaces.GPU
    def _warm_zerogpu():
        return None

    _warm_zerogpu()
except ImportError:
    pass

import api  # loads all four artifacts once, at startup

client = api.app.test_client()


def call_api(path, payload):
    # Send the JSON text ourselves: Flask's json= helper sorts the keys, and
    # a model whose scaler checks column order would then reject the request.
    response = client.post(path, data=json.dumps(payload), content_type="application/json")
    data = response.get_json(silent=True)
    if response.status_code != 200 or data is None:
        message = (data or {}).get("error") if isinstance(data, dict) else None
        raise gr.Error(message or f"The API returned HTTP {response.status_code}.")
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


def require(values, labels, optional=()):
    """Basic error handling: every non-optional field must be filled in."""
    missing = [
        label for label, value in zip(labels, values)
        if label not in optional and (value is None or (isinstance(value, str) and not value.strip()))
    ]
    if missing:
        raise gr.Error("Please fill in: " + ", ".join(missing))


# -----------------------------------------------------------------------
# 1. Regression -- monthly rent (KES) for a Nairobi-area property
# -----------------------------------------------------------------------
REG_CATEGORICAL = [
    ("property_type", "Apartment"),
    ("neighborhood", "Kilimani"),
    ("furnishing_status", "Fully Furnished"),
]
REG_NUMERIC = [
    ("property_age_years", 5.0),
    ("size_sqm", 100.0),
    ("num_bedrooms", 3),
    ("num_bathrooms", 2),
    ("distance_to_cbd_km", 8.0),
    ("service_charge_kes", 5000),
    ("garage_size_sqm", 15.0),
    ("security_rating", 4.0),
    ("amenities_score", 80.0),
    ("nearby_schools_count", 4),
]
REG_OPTIONAL = ("garage_size_sqm", "security_rating")
REG_FIELDS = [name for name, _ in REG_CATEGORICAL + REG_NUMERIC]


def predict_rent(*values):
    require(values, REG_FIELDS, REG_OPTIONAL)
    data = call_api(
        "/api/v1/models/property-rent-regressor/predictions",
        dict(zip(REG_FIELDS, values)),
    )
    return f"KES {data['predicted_monthly_rent_kes']:,.2f} / month\n\nModel: {data['model']}"


# -----------------------------------------------------------------------
# 2. Classification -- employee attrition
# -----------------------------------------------------------------------
CLF_CATEGORICAL = [
    ("department", "Operations"),
    ("job_role_level", "Junior"),
    ("education_level", "Diploma"),
    ("performance_rating", "Low"),
]
CLF_NUMERIC = [
    ("years_at_company", 1.5),
    ("years_since_last_promotion", 6.0),
    ("monthly_salary_kes", 25000),
    ("overtime_hours_per_month", 45.0),
    ("job_satisfaction_score", 1.5),
    ("work_life_balance_score", None),
    ("distance_from_home_km", 35.0),
    ("num_companies_worked_previously", 6),
    ("remote_work_days_per_week", 0),
    ("bonus_amount_kes", None),
]
CLF_OPTIONAL = ("work_life_balance_score", "bonus_amount_kes")
CLF_FIELDS = [name for name, _ in CLF_CATEGORICAL + CLF_NUMERIC]


def predict_attrition(*values):
    require(values, CLF_FIELDS, CLF_OPTIONAL)
    data = call_api(
        "/api/v1/models/employee-attrition-classifier/predictions",
        dict(zip(CLF_FIELDS, values)),
    )
    lines = [f"Will the employee leave? {data['predicted_attrition']}", ""]
    for label, prob in data.get("class_probabilities", {}).items():
        lines.append(f"  {label}: {prob:.1%}")
    lines += ["", f"Model: {data['model']}"]
    return "\n".join(lines)


# -----------------------------------------------------------------------
# 3. Clustering -- retail store segment
# -----------------------------------------------------------------------
CLU_NUMERIC = [
    ("monthly_revenue_kes", 620000),
    ("daily_foot_traffic", 610),
    ("staff_count", 7),
    ("store_size_sqm", 210),
    ("avg_basket_size_kes", 55),
]
CLU_FIELDS = [name for name, _ in CLU_NUMERIC]


def predict_segment(*values):
    require(values, CLU_FIELDS)
    data = call_api(
        "/api/v1/models/retail-store-segmenter/predictions",
        dict(zip(CLU_FIELDS, values)),
    )
    return f"Assigned segment: Cluster {data['assigned_cluster']}\n\nModel: {data['model']}"


# -----------------------------------------------------------------------
# 4. Association rules -- music track recommendations
# -----------------------------------------------------------------------
TRACK_CHOICES = [
    (f"{track_id} - {row['item_name']} ({row['category']})", track_id)
    for track_id, row in api.track_lookup.sort_index().iterrows()
]


def recommend_tracks(cart, top_n):
    if not cart:
        raise gr.Error("Pick at least one track for the listening session.")
    data = call_api(
        "/api/v1/models/music-recommender/predictions",
        {"cart": list(cart), "top_n": int(top_n or 3)},
    )
    if not data["recommendations"]:
        return "No recommendations for this session -- try e.g. M09 + M10, or M01 + M03."
    return "\n".join(
        f"{rec['track_name']} ({rec['track_id']}, {rec['category']})  --  "
        f"confidence {rec['confidence']:.1%}, lift {rec['lift']:.2f}"
        for rec in data["recommendations"]
    )


def text_input(pipeline, name, default):
    choices = categories_for(pipeline, name)
    label = name.replace("_", " ").capitalize()
    if choices:
        return gr.Dropdown(label=label, choices=choices,
                           value=default if default in choices else choices[0])
    return gr.Textbox(label=label, value=default)


def number_input(name, default, optional=()):
    label = name.replace("_", " ").capitalize()
    if name in optional:
        label += " (optional -- leave blank if unknown)"
    return gr.Number(label=label, value=default)


# -----------------------------------------------------------------------
# One gr.Blocks() app, four tabs -- one Space URL for all four models.
# -----------------------------------------------------------------------
with gr.Blocks(title="Group A2 Model Demos") as demo:
    gr.Markdown("# Group A2 -- Model Demos\nSwitch tabs to try a different model.")

    with gr.Tab("Property Rent Regressor"):
        gr.Interface(
            fn=predict_rent,
            inputs=[text_input(api.regression_pipeline, n, d) for n, d in REG_CATEGORICAL]
                   + [number_input(n, d, REG_OPTIONAL) for n, d in REG_NUMERIC],
            outputs=gr.Textbox(label="Predicted Monthly Rent"),
            flagging_mode="never",
        )

    with gr.Tab("Employee Attrition Classifier"):
        gr.Interface(
            fn=predict_attrition,
            inputs=[text_input(api.classification_pipeline, n, d) for n, d in CLF_CATEGORICAL]
                   + [number_input(n, d, CLF_OPTIONAL) for n, d in CLF_NUMERIC],
            outputs=gr.Textbox(label="Prediction"),
            flagging_mode="never",
        )

    with gr.Tab("Retail Store Segmenter"):
        gr.Interface(
            fn=predict_segment,
            inputs=[number_input(n, d) for n, d in CLU_NUMERIC],
            outputs=gr.Textbox(label="Prediction"),
            flagging_mode="never",
        )

    with gr.Tab("Music Recommender"):
        gr.Interface(
            fn=recommend_tracks,
            inputs=[
                gr.Dropdown(label="Tracks in the listening session", choices=TRACK_CHOICES,
                            value=["M09", "M10"], multiselect=True),
                gr.Number(label="Top N recommendations", value=3, precision=0),
            ],
            outputs=gr.Textbox(label="Recommendations"),
            flagging_mode="never",
        )

if __name__ == "__main__":
    demo.launch()
