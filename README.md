# Hotel Booking Cancellation Prediction

An end-to-end machine learning project for predicting whether a hotel reservation will be canceled. The project uses the Hotel Booking Demand dataset, compares six tree-based classifiers, performs randomized hyperparameter tuning, analyzes errors and classification thresholds, and exports the selected model as a Joblib artifact.

## Project Goal

Predict `is_canceled` using information that can reasonably be available when a reservation is made. The notebook deliberately excludes fields that describe what happened after the booking, reducing target leakage and making the model more appropriate for an operational booking-time use case.

## Repository Contents

```text
.
|-- data/
|   `-- hotel_bookings.csv              Input dataset
|-- hotel_booking_cancellation.ipynb   Complete analysis and training workflow
|-- hotel_booking_final_pipeline.joblib Trained LightGBM artifact
|-- catboost_info/                     CatBoost training logs
`-- README.md
```

The notebook is the source of truth for data preparation, experiments, evaluation, and model export. The checked-in `.joblib` file is the trained artifact produced by the notebook's final export cell.

## Dataset

The input file is `data/hotel_bookings.csv`, a tabular hotel-reservation dataset with 119,390 rows and 32 columns before cleaning. The target is:

- `is_canceled`: `0` for a reservation that was not canceled and `1` for a canceled reservation.

Important source fields include hotel, lead time, arrival date, stay length, guest counts, meal, country, market and distribution channels, previous booking history, room types, deposit type, customer type, average daily rate, parking, and special requests.

The initial target distribution is approximately 62.96% not canceled and 37.04% canceled. The data contains 31,994 exact duplicate rows, which the notebook removes before modeling. Missing values occur mainly in `company`, `agent`, `country`, and `children`.

## Methodology

### 1. Feature engineering

The notebook adds the following features:

- `total_nights`: weekend nights plus week nights.
- `total_guests`: adults plus children plus babies.
- `has_children`: whether at least one child is recorded.
- `has_babies`: whether at least one baby is recorded.
- `has_previous_cancellations`: whether the guest has canceled before.
- `adr_per_guest`: average daily rate divided by total guests.

The following columns are excluded from the feature matrix:

```text
is_canceled
reservation_status
reservation_status_date
assigned_room_type
booking_changes
```

`is_canceled` is the target. The remaining excluded fields may contain information generated after the original reservation decision or during later booking changes.

### 2. Train/test split

The cleaned data is split into training and test sets using an 80/20 stratified split with `random_state=42`. Exact duplicate removal leaves 87,396 rows, producing approximately 69,916 training rows and 17,480 test rows.

### 3. Preprocessing

For the scikit-learn, XGBoost, and LightGBM models:

- Numeric columns use median imputation.
- Categorical columns use most-frequent imputation followed by one-hot encoding.
- `OneHotEncoder(handle_unknown="ignore")` allows inference on previously unseen categories.

CatBoost receives categorical columns directly after missing categorical values are filled with `Unknown` and converted to strings.

### 4. Models

The baseline comparison includes:

1. Decision Tree using entropy splits
2. Random Forest
3. Gradient Boosting
4. XGBoost
5. LightGBM
6. CatBoost

Models are compared with accuracy, precision, recall, F1, ROC-AUC, PR-AUC, fitting time, and prediction time. ROC-AUC is used as the primary ranking metric during tuning.

### 5. Hyperparameter tuning

Each model is tuned with `RandomizedSearchCV`, a three-fold stratified cross-validation strategy, and ROC-AUC scoring. By default the notebook uses:

```python
QUICK_MODE = True
N_ITER = 5
```

Set `QUICK_MODE = False` to run 15 random configurations per model. This increases training time but gives a broader search.

## Results

The reported results below are from the executed notebook using the included dataset and `QUICK_MODE=True`.

### Baseline comparison

| Model             | Accuracy |       F1 |  ROC-AUC |   PR-AUC |
| ----------------- | -------: | -------: | -------: | -------: |
| LightGBM          | 0.844279 | 0.701862 | 0.910649 | 0.800956 |
| CatBoost          | 0.839416 | 0.685420 | 0.904880 | 0.789981 |
| XGBoost           | 0.839359 | 0.683070 | 0.902846 | 0.786475 |
| Gradient Boosting | 0.825572 | 0.634807 | 0.880812 | 0.742754 |
| Random Forest     | 0.798627 | 0.481896 | 0.871648 | 0.734839 |
| Decision Tree     | 0.807609 | 0.600641 | 0.862681 | 0.661086 |

### Tuned comparison

| Model             | Accuracy |       F1 |  ROC-AUC |   PR-AUC |
| ----------------- | -------: | -------: | -------: | -------: |
| LightGBM          | 0.847769 | 0.709149 | 0.913896 | 0.807064 |
| Random Forest     | 0.848284 | 0.709529 | 0.912096 | 0.803920 |
| XGBoost           | 0.841533 | 0.693923 | 0.907039 | 0.794511 |
| CatBoost          | 0.840046 | 0.688711 | 0.905963 | 0.791203 |
| Gradient Boosting | 0.825515 | 0.635516 | 0.881891 | 0.746580 |
| Decision Tree     | 0.798684 | 0.632019 | 0.792899 | 0.569769 |

LightGBM is selected by the notebook because it has the highest tuned test ROC-AUC and PR-AUC. Random Forest has a marginally higher accuracy and F1 in this run, so the final choice depends on the business objective and should not be based on ROC-AUC alone.

## Saved Model Artifact

`hotel_booking_final_pipeline.joblib` contains a dictionary with:

- `preprocessor`: the fitted `ColumnTransformer` containing numeric imputation and categorical one-hot encoding.
- `model`: the tuned LightGBM classifier.
- `feature_columns`: the feature columns expected before preprocessing.
- `target`: the target column name.
- `drop_columns`: columns excluded during training.

Because the artifact stores the fitted preprocessor and estimator together, new data must be passed through the same feature-engineering function and feature-column selection before calling `transform` and `predict_proba`.

Example inference code:

```python
import joblib
import pandas as pd

artifact = joblib.load("hotel_booking_final_pipeline.joblib")

new_bookings = pd.read_csv("new_bookings.csv")
new_bookings = engineer_features(new_bookings)  # use the function from the notebook
features = new_bookings.drop(columns=artifact["drop_columns"], errors="ignore")
features = features[artifact["feature_columns"]]

encoded = artifact["preprocessor"].transform(features)
cancel_probability = artifact["model"].predict_proba(encoded)[:, 1]
prediction = (cancel_probability >= 0.50).astype(int)
```

The `engineer_features` function is currently defined in the notebook rather than a standalone Python module, so production inference should extract that function into a reusable module before deployment.

## Running the Project

### Requirements

Python 3.9 or newer is recommended. Install the libraries used by the notebook:

```bash
python -m pip install numpy pandas matplotlib scikit-learn xgboost lightgbm catboost joblib jupyter
```

### Execute the notebook

From the repository root:

```bash
jupyter notebook hotel_booking_cancellation.ipynb
```

Run the cells from top to bottom. The notebook expects the dataset at `./data/hotel_bookings.csv` and writes the final artifact to the repository root.

## Evaluation and Business Use

The default decision threshold is `0.50`. The notebook evaluates thresholds from `0.10` through `0.90` and reports precision, recall, and F1 for each one. Lowering the threshold generally catches more likely cancellations but creates more false positives; raising it generally reduces false positives but misses more cancellations.

Hotels should choose the threshold according to the cost of overbooking, unnecessary intervention, lost inventory, and missed cancellation risk. The model output is a probability score, so it can support different operational policies instead of forcing one universal threshold.

The notebook also reports confusion matrices, classification reports, ROC curves, feature importances, and examples of false positives and false negatives for the selected model.

## Limitations and Next Steps

- The train/test split is random rather than time-based. A chronological holdout would better measure performance on future booking periods.
- The dataset includes historical booking patterns and may not represent every hotel's current customer mix or cancellation policy.
- Feature importance is model-specific and should not be interpreted as causality.
- The artifact does not include a standalone inference service or API.
- Re-run the full notebook after changing the dataset, feature policy, or tuning configuration.
- For deployment, package feature engineering and prediction into tested Python code, add input validation, monitor drift, and recalibrate the classification threshold using business costs.

## Project Workflow

```text
CSV data
	-> remove duplicates
	-> engineer booking-time features
	-> exclude leakage-prone columns
	-> stratified train/test split
	-> impute and one-hot encode (or pass categories to CatBoost)
	-> train six tree-based models
	-> compare baseline metrics
	-> tune with randomized 3-fold CV
	-> evaluate errors and thresholds
	-> select LightGBM by ROC-AUC
	-> save fitted preprocessing and model with Joblib
```
