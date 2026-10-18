# OTT Customer Analytics & Personalization

A local Flask API and React dashboard for customer analytics, existing segment
profiles, churn-risk estimates, and transparent genre suggestions.

## Architecture

- `app.py`: Flask API, cached customer/model/catalogue data, validation status,
  analytics, filtering, pagination, exports, and inference.
- `data_quality.py`: shared required schema and dataset validation helpers.
- `preprocessing.py`: consistent imputing/encoding/scaling for clustering input.
- `clustering.py`: saved K-Means assignments and plots.
- `churn_model.py`: reproducible churn model comparison, holdout evaluation,
  model persistence, and metadata/report generation.
- `frontend/`: React 19, Vite, Recharts, and the customer analytics UI.
- `data/ott_users.csv`: customer snapshot used by the application.
- `data/ott_users_preprocessed.csv` and `data/ott_users_clustered.csv`:
  existing preprocessing output and cluster assignments.

The API uses `http://127.0.0.1:5000`; the Vite development UI uses
`http://localhost:5173`. CORS permits those local frontend origins by default.
Set `CORS_ORIGINS` to a comma-separated allowlist when using a different
frontend origin. Do not expose the Flask development server directly to the
internet.

## Install and run (Windows PowerShell)

Create/use a Python 3.11 virtual environment, then install the backend
dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

In terminal 1, start the API:

```powershell
.\.venv\Scripts\python.exe app.py
```

In terminal 2, install and start the frontend:

```powershell
cd frontend
npm ci
npm run dev
```

Open `http://localhost:5173`. Use the refresh action to reload dashboard data.
`POST /api/reload` reloads customer data, saved cluster assignments, the model,
metadata, and an optional movie catalogue without restarting Flask. A failed
customer validation does not replace the currently loaded valid customer
snapshot.

## Data schema and quality

The required `data/ott_users.csv` schema is:

| Column | Meaning in the available file |
| --- | --- |
| `customer_id` | Customer key; must be present and unique for unambiguous detail views |
| `age` | Age in years |
| `gender` | `Female`, `Male`, or `Other` |
| `subscription_type` | `Basic`, `Standard`, or `Premium` |
| `watch_hours` | Recorded viewing hours; source time window is not documented |
| `last_login_days` | Days since last login in this snapshot |
| `region` | Recorded customer region |
| `device` | Recorded device category |
| `monthly_fee` | Subscription fee; currency is not specified |
| `churned` | Binary historical label, 0 or 1; label definition/horizon is not supplied |
| `payment_method` | Recorded payment category |
| `number_of_profiles` | Profile count |
| `avg_watch_time_per_day` | Average viewing hours per day |
| `favorite_genre` | One recorded favourite genre |

The current CSV has 5,000 rows, 14 required columns, no missing cells, and no
duplicate customer IDs in the inspected copy. Validation checks required
columns, numeric convertibility, missing values, duplicate IDs, observed
category vocabulary, and safe domain ranges. The dashboard shows a quality
summary; `/api/ready` reports readiness and validation details.

**Dataset source and limitations:** no dataset source, collection date,
generation process, churn-label definition, or observation window is documented
in the repository. Its provenance is therefore unknown; do not assume that it
represents production customers. No usable event/observation timestamp,
customer-item interaction history, language, maturity rating, or movie
catalogue was present at inspection. Viewing-hour boundaries and the currency
for `monthly_fee` are also unspecified. The existing CSV files are part of the
provided repository; do not add private customer exports or credentials.

The dashboard defines “active” as `last_login_days <= 7`. Viewing and
inactivity bands are descriptive groupings, not proof of causal relationships.
Date filters and time-based model evaluation are not available without reliable
timestamps.

## Churn model and reproducibility

Train and evaluate with the same feature schema used for inference:

```powershell
.\.venv\Scripts\python.exe churn_model.py
```

The command uses a stratified 80/20 random split (`random_state=42`) and
compares a majority-class baseline, Logistic Regression, Gradient Boosting, and
Random Forest. It reports accuracy, precision, recall, F1, ROC-AUC, PR-AUC,
confusion counts, class distribution, and threshold tradeoffs at 0.25, 0.50,
and 0.75. The Random Forest remains the persisted model to preserve the
existing inference type. Outputs:

- `churn_model.pkl`: trained pipeline, ignored by Git as a generated binary.
- `churn_model_metadata.json`: model version, training date, schema,
  evaluation, threshold explanation, and global feature importance.
- `reports/churn_evaluation.json`: complete machine-readable comparison.

The target and customer identifier are excluded from predictors. The input
columns are treated as available at scoring time, but their freshness cannot be
verified without snapshot timestamps. A chronological split is unavailable.
Feature importance is global to the fitted model; it is not a causal or
customer-specific explanation. No SHAP dependency is installed.

The API defaults to a 0.50 classification threshold; the Customers page
provides an adjustable threshold. Lower thresholds generally increase
recall and false positives; higher thresholds generally reduce false positives
but miss more churners. Risk bands are low `<33%`, medium `33–65.9%`, and high
`>=66%`, independent of the classification threshold. Risk is an estimate and
does not mean a customer will definitely churn.

## Segmentation

The UI uses the existing saved cluster assignment file and shows size, viewing,
recency, churn label rate, plan mix, and favourite-genre mix. Names summarize
viewing and recency relative to overall medians; original cluster IDs are
retained. The Segments page compares multiple existing segments and evaluates
candidate `k=2..8` on the existing preprocessed customer features. The
customer identifier is excluded. Inertia (lower) and silhouette (higher) are
shown as complementary descriptive criteria, not an automated causal or
business decision.

Run `preprocessing.py` before `clustering.py` to regenerate the feature matrix
and cluster assignments. The current checked-in cluster labels were generated
by the project scripts; no training timestamp or cluster-model metadata is
provided.

## Recommendations

The available customer data supports favourite-genre suggestions only. Those
suggestions use an explicit fixed mapping and are labelled as a transparent
baseline, not a learned recommender. There is no interaction history for
Precision@K, Recall@K, NDCG@K, coverage, or diversity evaluation.

The supplied `data/movies_metadata.csv` catalogue contains `title`, a serialized
`genres` list, `overview`, `original_language`, `release_date`, and other movie
metadata. The API parses its genre names and presents them as catalogue matches
without requiring manual conversion. Titles match the customer's favourite
genre case-insensitively; up to five matches are returned in title order with
available synopsis, language, year, rating-count, popularity, and runtime
metadata. Genre and language filters are enabled from the catalogue, as is a
release-year filter derived from `release_date`. The customer page identifies
the file as the source and warns that catalogue metadata does not establish
current streaming availability.

The catalogue was supplied in the project workspace; its upstream provenance
and licensing are not documented here. The file may describe a third-party
movie metadata snapshot, not titles available on a particular OTT service.
There is no customer-item interaction history, so these are exact genre
catalogue matches, not learned or evaluated personalized recommendations. The
genre-suggestion baseline remains separate. The API does not create titles or
metadata.

Retention suggestions are based on observable viewing/inactivity percentiles.
They are editable informational notes only; the application does not contact
customers or apply offers.

## API

| Method and path | Purpose |
| --- | --- |
| `GET /api/health` | Process liveness |
| `GET /api/ready` | Data/model/catalogue readiness and validation |
| `POST /api/reload` | Reload data and models without restarting |
| `GET /api/dashboard` | KPIs, chart series, and data-quality summary |
| `GET /api/customers` | Filtered, sortable, paginated customer records |
| `GET /api/customers/export.csv` | CSV of the current filters plus summary rows |
| `GET /api/customer/<customer_id>` | Customer profile, activity, segment, suggestions, and risk |
| `GET /api/churn/<customer_id>?threshold=0.5` | Risk score, classification, and global feature importance |
| `GET /api/model` | Model availability, version, schema, and evaluation metadata |
| `GET /api/segments` | Existing segment profiles |
| `GET /api/segments/evaluation` | Candidate `k`, inertia, and silhouette |
| `GET /api/recommendations` | Genre-level baseline suggestions and explanations |
| `GET /api/movie-recommendations/<customer_id>` | Exact favourite-genre matches from `data/movies_metadata.csv` |

Customer filters include `q`, `segment`, `churned`, `risk_band`,
`subscription_type`, `region`, `device`, `favorite_genre`, `inactivity_band`,
and `watch_band`. Pagination uses `page` and `page_size` (10, 25, 50, or 100).
Sorting uses `sort_by` and `order`. Invalid request parameters return a
consistent JSON error with an error code.

## Tests and build

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
cd frontend
npm test
npm run lint
npm run build
```

The frontend test suite uses Node's built-in test runner. Build output may
include a Vite bundle-size advisory from Recharts; it does not prevent a
successful build


