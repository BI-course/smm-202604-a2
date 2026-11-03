"""
api.py

A RESTish API, built with Flask, that serves FOUR machine learning artifacts
produced in earlier labs:

    1. Regression       -- SME monthly revenue prediction
    2. Classification   -- SME credit risk category
    3. Clustering       -- Retail store segment assignment
    4. Association Rule -- Product recommendation from a cart
       Mining (ARM)

The point of this file is to show you that a fitted scikit-learn /
imbalanced-learn Pipeline is a SINGLE object that already knows how to impute
missing values, encode categorical columns, scale numeric columns, and (for
the classifier) apply SMOTE only during training -- never during inference.
Because all of that preprocessing lives inside the Pipeline itself, this file
never manually label-encodes anything, never manually engineers date features,
and never manually reorders columns before calling .predict(). It builds a
DataFrame straight from the incoming JSON and hands it to the pipeline as-is.
This is the main lesson of the lab: "the pipeline IS the deployment artifact."

Run this file directly to start a local development server:

    python api.py
"""

import os

import joblib
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)

# -----------------------------------------------------------------------
# CORS (Cross-Origin Resource Sharing)
# -----------------------------------------------------------------------
# By default, a browser blocks JavaScript on one origin (e.g., a page served
# from http://127.0.0.1:5500) from calling an API on a different origin
# (e.g., http://127.0.0.1:5000). `flask_cors` relaxes that restriction for the
# specific origins we trust during local development.
#
# A few alternative configurations are shown below (commented out) so you can
# see how the restriction can be loosened or tightened depending on your
# deployment target. Only ONE of these should be active at a time.

# Option A (the most permissive. This is fine for a quick classroom demo or
# local development, *NEVER* for production):
# CORS(app)

# Option B (allow only a specific frontend origin):
# CORS(app, resources={r"/api/*": {"origins": "http://127.0.0.1:5500"}})

# Option C (the one actually in use here -- allow localhost/127.0.0.1 on the
# ports our lab's frontend files are typically served from):
CORS(app, resources={r"/api/*": {"origins": [
    "http://localhost:5000", "http://127.0.0.1:5000",
    "http://localhost:5500", "http://127.0.0.1:5500",
    "https://localhost:443", "https://127.0.0.1:443",
    "http://localhost:63342",  # VS Code Live Server extension
    "null",  # allows opening a frontend .html file directly (file://) with no web server
]}})

# -----------------------------------------------------------------------
# The best practice is to load the four persisted artifacts ONCE, at startup,
# not per-request.
# Loading inside a route function would re-read the file from disk on every
# single API call, which is slow and pointless -- the fitted model never
# changes while the server is running.
# -----------------------------------------------------------------------
MODEL_DIR = "./model"

regression_pipeline = joblib.load(
    os.path.join(MODEL_DIR, "elasticnet_regressor_for_property_rent.joblib")
)

# Our group's tuned Logistic Regression pipeline for employee attrition
# (preprocessing + SMOTE + model), saved in the Classification lab.
# It is loaded once at startup -- never retrained here.
classification_pipeline = joblib.load(
    os.path.join(MODEL_DIR, "employee_attrition_model.pkl")
)


clustering_artifact = joblib.load(
    os.path.join(MODEL_DIR, "kmeans_retail_store_segmentation.joblib")
)

clustering_imputer = clustering_artifact["imputer"]
clustering_scaler = clustering_artifact["scaler"]
clustering_kmeans = clustering_artifact["kmeans"]

clustering_features = [
    "monthly_revenue_kes",
    "daily_foot_traffic",
    "staff_count",
    "store_size_sqm",
    "avg_basket_size_kes"
]

arm_artifact = joblib.load(
    os.path.join(MODEL_DIR, "apriori_music_recommendation_rules.joblib")
)
recommendation_lookup = arm_artifact["recommendation_lookup"]
track_lookup = arm_artifact["track_lookup"]

print("All the four artifacts have been loaded. The API is ready.")


# 1. REGRESSION -- predict monthly rent (KES) for a Nairobi-area property

# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/property-rent-regressor/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"property_type": "Apartment", "neighborhood": "Kilimani",
#             "furnishing_status": "Fully Furnished",
#             "property_age_years": 5.0, "size_sqm": 100.0,
#             "num_bedrooms": 3, "num_bathrooms": 2,
#             "distance_to_cbd_km": 8.0, "service_charge_kes": 5000,
#             "garage_size_sqm": 15.0, "security_rating": 4.0,
#             "amenities_score": 80.0, "nearby_schools_count": 4}'
#
# Expected response (matches the regression notebook):
#   {"predicted_monthly_rent_kes": 125400.21, ...}
#
# Note: garage_size_sqm and security_rating may be sent as null. The
# pipeline's median imputer handles them; this route needs no special code.
@app.route("/api/v1/models/property-rent-regressor/predictions", methods=["POST"])
def predict_rent():
    payload = request.get_json()

    # One-row DataFrame straight from the JSON body -- the pipeline's
    # ColumnTransformer handles imputation, scaling and one-hot encoding.
    input_row = pd.DataFrame([payload])

    predicted_log_rent = regression_pipeline.predict(input_row)[0]
    # The pipeline was trained on log1p(rent); invert it to report KES.
    predicted_rent_kes = float(np.expm1(predicted_log_rent))

    return jsonify({
        "predicted_monthly_rent_kes": round(predicted_rent_kes, 2),
        "model": "ElasticNet Regressor (tuned, alpha=0.001, l1_ratio=0.8)",
    })


# 2. CLASSIFICATION -- predict employee attrition (Yes = likely to leave, No = likely to stay)

# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/sme-credit-risk-classifier/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"department": "Operations", "job_role_level": "Junior",
#             "education_level": "Diploma", "performance_rating": "Low",
#             "years_at_company": 1.5, "years_since_last_promotion": 6.0,
#             "monthly_salary_kes": 25000, "overtime_hours_per_month": 45.0,
#             "job_satisfaction_score": 1.5, "work_life_balance_score": null,
#             "distance_from_home_km": 35.0, "num_companies_worked_previously": 6,
#             "remote_work_days_per_week": 0, "bonus_amount_kes": null}'
#
# Note: missing values are sent as null (e.g. an employee with no bonus). The
# pipeline's own median imputer fills them in, so this route needs no special
# code for missing data. The ordinal columns only accept the categories the
# model was trained on:
#   education_level:    Diploma, Bachelor's, Master's, PhD
#   performance_rating: Low, Average, High, Outstanding
@app.route("/api/v1/models/sme-credit-risk-classifier/predictions", methods=["POST"])
def predict_employee_attrition():
    # silent=True returns None instead of crashing if the body is not valid JSON
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Send the employee's details as a JSON body."}), 400

    # Build a one-row DataFrame straight from the JSON body. The pipeline's
    # ColumnTransformer already knows how to impute, scale, one-hot encode and
    # ordinal-encode each column.
    input_row = pd.DataFrame([payload])

    try:
        predicted_class = classification_pipeline.predict(input_row)[0]
        predicted_probabilities = classification_pipeline.predict_proba(input_row)[0]
    except (ValueError, KeyError) as error:
        # e.g. a missing column, or an education_level/performance_rating value
        # the model never saw during training
        return jsonify({"error": f"Invalid input: {error}"}), 400

    # The labels the model was trained on, in the same order as the probabilities: ["No", "Yes"]
    class_labels = classification_pipeline.classes_

    return jsonify({
        "predicted_attrition": str(predicted_class),
        "class_probabilities": {
            str(label): round(float(prob), 4)
            for label, prob in zip(class_labels, predicted_probabilities)
        },
        "model": "Logistic Regression (tuned, SMOTE-resampled at training only)",
    })


# 3. CLUSTERING -- assign a new retail store to an existing segment
#
#  Sample request (cURL):
#    curl -X POST http://127.0.0.1:5000/api/v1/models/retail-store-segmenter/predictions \
#         -H "Content-Type: application/json" \
#         -d '{"monthly_revenue_kes": 620000, "daily_foot_traffic": 610, "staff_count": 7, "store_size_sqm": 210, "avg_basket_size_kes": 55}'
#
#  The imputer handles missing values, the scaler standardizes the
#  five numeric features, and K-Means assigns the store to a cluster.
#  These are the same preprocessing and clustering objects saved
# together in the Retail Store Segmentation notebook.

@app.route("/api/v1/models/retail-store-segmenter/predictions", methods=["POST"])
def predict_store_segment():
     payload = request.get_json(silent=True)

     if not isinstance(payload, dict):
         return jsonify({"error": "Send store details as a JSON object."}), 400

     missing_features = [
         feature for feature in clustering_features
         if feature not in payload
     ]

     if missing_features:
         return jsonify({
             "error": "Missing required features.",
             "missing_features": missing_features
         }), 400

     try:
         input_row = pd.DataFrame(
             [[payload[feature] for feature in clustering_features]],
             columns=clustering_features
         )

         imputed_input = clustering_imputer.transform(input_row)
         scaled_input = clustering_scaler.transform(imputed_input)

         assigned_cluster = int(
             clustering_kmeans.predict(scaled_input)[0]
         )

     except (ValueError, TypeError) as error:
         return jsonify({"error": f"Invalid input: {error}"}), 400

     return jsonify({
         "assigned_cluster": assigned_cluster,
         "model": "K-Means Retail Store Segmentation (k=5)"
     })

# 4. ASSOCIATION RULE MINING -- recommend products for a shopping cart

# Sample request (cURL):
#   curl -X POST http://127.0.0.1:5000/api/v1/models/ecommerce-recommender/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"cart": ["H06", "H07"]}'
#
#   curl -X POST http://127.0.0.1:5000/api/v1/models/ecommerce-recommender/predictions \
#        -H "Content-Type: application/json" \
#        -d '{"cart": ["P04", "P01"]}'
#
# Note: there is no scikit-learn ".predict()" here -- the mined association
# rules were precomputed into a lookup dictionary
# (antecedent SKU-set -> ranked candidate consequents)
# during the notebook's persistence stage. This
# endpoint performs the exact same lookup+rank step a production recommender
# microservice would perform on every incoming request; it does not mine
# for the rules again.
@app.route("/api/v1/models/ecommerce-recommender/predictions", methods=["POST"])
def recommend_products():
    payload = request.get_json(silent=True)
    if not payload or not isinstance(payload.get("cart"), list) or not payload["cart"]:
        return jsonify({"error": "Provide a non-empty list of track IDs under 'cart'"}), 400
    cart = set(payload["cart"])
    try:
        top_n = int(payload.get("top_n", 3))
    except (ValueError, TypeError):
        return jsonify({"error": "'top_n' must be an integer"}), 400

    candidates = {}
    for antecedent, consequent_list in recommendation_lookup.items():
        if antecedent.issubset(cart):
            for track_id, confidence, lift in consequent_list:
                if track_id not in cart:
                    if track_id not in candidates or confidence > candidates[track_id][0]:
                        candidates[track_id] = (confidence, lift)

    ranked = sorted(candidates.items(), key=lambda x: x[1][0], reverse=True)[:top_n]
    recommendations = []
    for track_id, (confidence, lift) in ranked:
        known = track_id in track_lookup.index
        recommendations.append({
            "track_id": str(track_id),
            "track_name": track_lookup.loc[track_id, "item_name"] if known else None,
            "category": track_lookup.loc[track_id, "category"] if known else None,
            "confidence": round(float(confidence), 3),
            "lift": round(float(lift), 2),
        })

    return jsonify({
        "cart": sorted(cart),
        "recommendations": recommendations,
        "model": "Apriori association rules (music), precomputed lookup",
    })

# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------
if __name__ == "__main__":
    # Development server. debug=True gives auto-reload and an interactive
    # debugger on errors -- convenient in class, but NEVER use debug=True in
    # a deployed/public environment (it can expose a remote code execution
    # console to anyone who can reach the server).
    app.run(debug=True)

    # Production-style alternative (no debugger, no auto-reload):
    # app.run(debug=False, host="0.0.0.0", port=5000)

    # HTTPS/TLS alternative for local development (requires a self-signed
    # certificate; see app_server_reverse_proxy_server_setup.md):
    # app.run(debug=False, ssl_context=("cert.pem", "key.pem"))
