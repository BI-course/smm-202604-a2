# Lab Submission Instructions

## Student Details and Individual Member Contributions

**Name of the team on GitHub Classroom:** A2

**Member 1:**

| **Details**                                                                                                                           | **Comment**             |
|:--------------------------------------------------------------------------------------------------------------------------------------|:------------------------|
| **Student ID**                                                                                                                        | 161136                  |
| **Name**                                                                                                                              | Abdullahi Samiya Issack |
| **What part of the lab did you personally<br/>contribute to (provide a link to the<br/>branch(es)), and what did you learn from it?** |         **Branches:** [samiya-regression](https://github.com/BI-course/smm-202604-a2/tree/samiya-regression) (regression endpoint) and [main](https://github.com/BI-course/smm-202604-a2/tree/main) (merging, deployment and Advanced-tier apps).<br/><br/>**What I did:** Served our group's tuned ElasticNet property-rent model through the regression endpoint and confirmed it matches the notebook's prediction (KES 125,400.21). Fixed the scikit-learn version mismatch (models saved with 1.6.1) by pinning it for every deployment. Merged all members' branches into `main`, testing all four endpoints after each merge. Got Docker + Gunicorn + Nginx working by running both compose files together and generating the self-signed certificate. Deployed the API to Render. For the Advanced tier, deployed the Gradio app to Hugging Face Spaces and the Streamlit app to Streamlit Community Cloud (one tab per model, both reusing the same `api.py` and model files), and added an HTML/CSS/JS page with error handling for missing input.<br/><br/>**What I learned:** A saved scikit-learn Pipeline carries its own preprocessing, so the API only has to reverse the log transform. Models must be loaded with the same scikit-learn version they were saved with, and every platform needs that version pinned. Free platforms also pick their own Python version (Streamlit chose 3.14, which had no ready-made scikit-learn 1.6.1), so it has to be set explicitly. Reusing one `api.py` everywhere keeps local, Docker, Render, Hugging Face and Streamlit giving identical answers.                 |

**Member 2:**

| **Details**                                                                                                                           | **Comment** |
|:--------------------------------------------------------------------------------------------------------------------------------------|:------------|
| **Student ID**                                                                                                                |  166437             |
| **Name**                                                                                                             |     Fatuma Mohamed           |
| **What part of the lab did you personally<br/>contribute to (provide a link to the<br/>branch(es)), and what did you learn from it?** |     **Branch:** [fatuma-classification](https://github.com/BI-course/smm-202604-a2/tree/fatuma-classification)<br/><br/>**What I did:** I handled the classification part. I took our employee attrition model (the tuned Logistic Regression from our classification lab) and added it to `api.py`, so you can send an employee's details and it tells you whether they're likely to leave (Yes or No) and how sure it is. I renamed the route to `employee-attrition-classifier` because the old name was still from the lecturer's example. I also added checks so that if someone sends nothing, or types a category the model doesn't know, it gives a clear error message instead of crashing. Apart from my endpoint, I made the test script (`test_requests.sh`) and the Postman collection that the whole group used to test all four endpoints locally, in Docker and on Render.<br/><br/>**What I learned:** I didn't realise how much the saved pipeline already does. All the cleaning and encoding we did in the notebook is saved inside it, so the API just passes the data in. I also understood properly why SMOTE is only used during training and not when making predictions. The error handling taught me that a user can easily send a value the model has never seen, so the API has to expect that. And having one test script saved us a lot of time, because we could run the same tests every time something changed.        |

**Member 3:**

| **Details**                                                                                                                           | **Comment** |
|:--------------------------------------------------------------------------------------------------------------------------------------|:------------|
| **Student ID**                                                                                                                        |        153527            |
| **Name**       Veronicah Mbuvi                                                                                                        |          Veronicah Mbuvi       |
| **What part of the lab did you personally<br/>contribute to (provide a link to the<br/>branch(es)), and what did you learn from it?** | https://github.com/BI-course/smm-202604-a2/tree/veronicah-clustering I handled the Retail Store Segmentation part of the lab. I integrated our trained K-Means retail store segmentation model into api.py so that users can submit details about a retail store and receive the cluster it belongs to. I laso loaded the saved model together with its preprocessing components, including the imputer and scaler. I created a POST endpoint  that accepts five features: monthly revenue, daily foot traffic, staff count, store size and average basket size. I also added input validation to check for missing or invalid values and ensured that the API returns clear error messages instead of crashing. The endpoint returns the assigned cluster in JSON format.I learned how a trained machine learning model can be integrated into a Flask API and made accessible to other applications. I learned how to create POST endpoints, validate incoming JSON data and handle errors. 
            |
**Member 4:**

| **Details**                                                                                                                           | **Comment** |
|:--------------------------------------------------------------------------------------------------------------------------------------|:------------|
| **Student ID**                                                                                                                        |             |
| **Name**                                                                                                                              |             |
| **What part of the lab did you personally<br/>contribute to (provide a link to the<br/>branch(es)), and what did you learn from it?** |             |

**Member 5:**

| **Details**                                                                                                                           | **Comment** |
|:--------------------------------------------------------------------------------------------------------------------------------------|:------------|
| **Student ID**                                                                                                                        |             |
| **Name**                                                                                                                              |             |
| **What part of the lab did you personally<br/>contribute to (provide a link to the<br/>branch(es)), and what did you learn from it?** |             |

## Chosen Level of Difficulty

**Specify the chosen level of difficulty** (baseline, intermediate, or advanced): Advanced

## Video Demonstration

Submit the link to a short video (**not more than 10 minutes**) demonstrating your complete solution at whichever difficulty level you chose. Please ensure that the lecturer has rights to view the video.

At minimum, the video must show, in one continuous walkthrough:

1. `api.py` running **locally**, with a live request/response against each model endpoint you implemented.
2. If you attempted the Intermediate or Advanced tier: the same API running inside its **Gunicorn + Nginx Docker container**, reached through the reverse proxy, not the Flask development server.
3. If you attempted the Advanced tier: your **Hugging Face Space** and your **Streamlit Community Cloud app**, each opened live in a browser, with at least one prediction made on each tab, and your **Render**-hosted API answering a request from Postman or `curl`.

**Why this matters for grading:** free-tier public deployments are not guaranteed to still exist by the time this is graded. The video is treated as the authoritative record of what was actually working at submission time; a broken link at grading time is not penalized if the video clearly shows that deployment working.

Note that you are required to submit the link to the video and NOT the video itself. The video should NOT be uploaded to your repository—that would be a misuse of GitHub.

**Link to the video:**
https://youtu.be/JmS7e-HJIrU

## Public URLs

Fill in only the rows that apply to your chosen difficulty level (see Part C below). Leave a row blank if your tier does not require that deployment target.

| **Deployment target**                             | **URL** |
|:--------------------------------------------------|:--------|
| Render (the full `api.py`, all endpoints)         |     https://smm-202604-a2-api.onrender.com<br/>POST `/api/v1/models/property-rent-regressor/predictions`<br/>POST `/api/v1/models/employee-attrition-classifier/predictions`<br/>POST `/api/v1/models/retail-store-segmenter/predictions`<br/>POST `/api/v1/models/music-recommender/predictions`    |
| Hugging Face Space (Gradio, one tab per model)    | https://huggingface.co/spaces/Samiwami/smm-a2-models        |
| Streamlit Community Cloud app (one tab per model) | https://202604-smm-a2.streamlit.app/        |

---

## Scenario

Across earlier labs, **your group** was assigned its own dataset and trained your own models for four distinct tasks:

1. A **regression** task
2. A **classification** task
3. A **k-Means clustering** task
4. An **Apriori association-rule mining** task

Different groups worked with different datasets and, in some cases, different algorithms for each task -- there is no single shared reference repository or model list for the whole class. Wherever this document refers to "your regressor," "your classifier," "your clustering model," or "your recommender," it means **whichever model your own group already trained and saved for that task**, not a specific named algorithm.

You are now required to build a single Flask API (`api.py`) that serves **your own group's four models**, loaded from disk (never retrained inside `api.py`), and to demonstrate that API running correctly across a growing number of deployment surfaces depending on the difficulty tier you choose.

### Part A: Regression and Classification

- Locate the trained regression and classification model files your group produced and saved in your earlier labs on Regression and Classification.
- Update [api.py](api.py) to include end-points that load these two models from disk and serve predictions from them.

### Part B: Clustering and Association Rule Mining

- Locate the trained clustering model and the association rules your group produced and saved in your earlier labs on Clustering and Association Rule Mining.
- Update [api.py](api.py) to include end-points that:
  - Load the clustering model from disk and predict which cluster a new client belongs to
  - Load the association rules from disk and recommend products based on them

**Note 1:** **`api.py` is NOT production-grade as it is.** It is only meant for demonstration purposes. Scalability and security must be taken into consideration before deploying an API in a production environment.

**Note 2:** Some students often treat the API as an afterthought, focusing only on training ML models. In practice, **the API is the product** -- it is how others interact with your model. The "hidden" learning here is that the delivery mechanism (API design, usability, error handling, and even documentation) often matters more to stakeholders in the industry than the models themselves.

### Part C: Deployment Breadth (Read Before Choosing Your Tier)

All three tiers below ask you to serve **the same four models** (your regressor, your classifier, your clustering model, your recommender) from **one, unmodified `api.py`**. The model files and the Flask routes do not change between tiers -- only how many different places that one working API is shown to run correctly.

**Baseline (Required):**

- Update `api.py` to serve **at least three of your four** models, loaded from disk.
- Run and demonstrate `api.py` **locally** with Flask's development server, with a successful request against each implemented endpoint.

**Intermediate (Recommended):**

- Update `api.py` to serve **all four** of your models, loaded from disk.
- Dockerize your Flask API using **Gunicorn**, and place it behind an **Nginx** reverse proxy (see the reverse-proxy lab notes for the full Docker Compose setup).
- Deploy the same, unmodified `api.py` to **Render**, and demonstrate a real request against it from outside your own machine (cURL or Postman).

**Advanced (Optional):**

- Everything required for Intermediate, plus:
  - Create a web page (or pages) using Basic HTML, CSS, and Vanilla JavaScript that demonstrates the use of the API
  - Implement basic error handling (e.g., missing inputs)
  - Build **one Gradio app with a tab per model** (all four of your models) and publish it as a **Hugging Face Space**
  - Build **one Streamlit app with a tab per model**, covering the same four models, and publish it on **Streamlit Community Cloud**
  - All deployment targets -- local, Dockerized/Nginx, Render, Hugging Face, and Streamlit -- must serve models loaded **from the same trained artifacts**; retraining different models per platform does not satisfy this requirement

Why this is production-relevant beyond the model itself:

- **Portability:** the same container image and the same trained artifacts run unmodified on your laptop, inside Docker, and on a managed cloud host.
- **Consistency:** no "but it works on my machine" gap between what you demo locally and what a domain expert sees on a public URL.
- **Separation of concerns:** a REST API (Render), a no-code demo UI (Hugging Face/Streamlit), and your own frontend (the Advanced-tier HTML pages) are three different ways of exposing the *same* underlying models to three different kinds of user -- a developer integrating your API, a non-technical domain expert validating it, and an end customer.

---

## Marking Guide

Refer to the marking guide available here: [lab_submission_marking_guide.md](lab_submission_marking_guide.md)
