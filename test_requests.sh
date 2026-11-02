#!/usr/bin/env bash
# =============================================================================
# test_requests.sh -- one working request per endpoint, reused at every stage.
#
# The base URL is a variable, so the SAME requests run against:
#   Local Flask (python api.py):  bash test_requests.sh http://127.0.0.1:5000
#   Docker via Nginx (HTTPS):     bash test_requests.sh https://localhost
#   Render:                       bash test_requests.sh https://<our-app>.onrender.com
#
# Notes:
#   - curl's -k flag accepts Nginx's self-signed certificate.
#   - Render's free tier sleeps; the first request can take about a minute.
#   - If an endpoint's input fields change in api.py, update its JSON below
#     (and the Postman collection) so both stay in sync.
# =============================================================================

# Use the first argument, else the BASE_URL variable, else local Flask
BASE_URL="${1:-${BASE_URL:-http://127.0.0.1:5000}}"
# Remove a trailing slash so URLs don't end up with "//api"
BASE_URL="${BASE_URL%/}"

echo "Testing API at: ${BASE_URL}"

FAILED=0

# Sends one POST request and prints the response and HTTP status code.
#   $1 = name shown in the output, $2 = endpoint path, $3 = JSON body
call() {
  local name="$1" path="$2" body="$3"
  echo
  echo "=== ${name} ==="
  echo "POST ${BASE_URL}${path}"
  local response status
  # -w appends the HTTP status code on its own last line
  response=$(curl -sk --max-time 120 -w '\n%{http_code}' -X POST "${BASE_URL}${path}" \
    -H "Content-Type: application/json" -d "${body}")
  status="${response##*$'\n'}"
  echo "${response%$'\n'*}"
  echo "HTTP ${status}"
  [ "${status}" = "200" ] || FAILED=$((FAILED + 1))
}

# 1. Regression (Member 1) -- replace with Member 1's sample JSON
call "Regression" "/api/v1/models/sme-revenue-regressor/predictions" '{
  "sector": "Technology", "county": "Nairobi", "quarter": "Q3",
  "business_age_years": 4.5, "num_employees": 12,
  "owner_education_level": "Secondary",
  "monthly_marketing_spend_kes": 45000,
  "avg_daily_foot_traffic": null,
  "online_presence_score": 78.0, "loan_amount_kes": 250000,
  "interest_rate_pct": 14.5, "inflation_rate_pct": 6.5,
  "competitor_density": 5, "customer_satisfaction_score": 4.2
}'

# 2. Classification (Member 2) -- employee attrition
call "Classification" "/api/v1/models/employee-attrition-classifier/predictions" '{
  "department": "Operations", "job_role_level": "Junior",
  "education_level": "Diploma", "performance_rating": "Low",
  "years_at_company": 1.5, "years_since_last_promotion": 6.0,
  "monthly_salary_kes": 25000, "overtime_hours_per_month": 45.0,
  "job_satisfaction_score": 1.5, "work_life_balance_score": null,
  "distance_from_home_km": 35.0, "num_companies_worked_previously": 6,
  "remote_work_days_per_week": 0, "bonus_amount_kes": null
}'

# 3. Clustering (Member 3) -- replace with Member 3's sample JSON
call "Clustering" "/api/v1/models/mall-customer-segmenter/predictions" '{
  "Age": 27, "Annual Income (k$)": 45, "Spending Score (1-100)": 72
}'

# 4. Recommender / association rules (Member 4) -- replace with Member 4's sample JSON
call "Recommender" "/api/v1/models/ecommerce-recommender/predictions" '{
  "cart": ["H06", "H07"]
}'

# Summary: exit code 1 if any endpoint failed, so problems are obvious
echo
if [ "${FAILED}" -eq 0 ]; then
  echo "All 4 endpoints returned HTTP 200."
else
  echo "${FAILED} endpoint(s) did NOT return HTTP 200."
  exit 1
fi