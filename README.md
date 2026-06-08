# Steel Client Purchase Prediction

## Project Overview

This project analyzes client behavior for a steel manufacturing company using the **STEEL CLIENTS** dataset. The goal is to predict `NUMBER_OF_PURCHASES` based on client web activity, negotiations, tonnage behavior, and platform usage.

## Business Problem

Alpha Steel implemented a web-based order-to-purchase system. This project evaluates how client behavior on the platform can be used to predict purchase activity and support digital sales decision-making.

## Target Variable

`NUMBER_OF_PURCHASES`

## Features Used

- Number of negotiations
- Sessions per year
- Tons confirmed
- Sessions on web app
- Distribution center changes
- Delivery or pickup preference
- Executive-assisted sessions
- Average actions
- Pages visited
- Excel tool usage
- Client catalogue usage
- Client classification

## Models Compared

- OLS Regression
- Ridge Regression
- Lasso Regression
- Polynomial Regression
- Neural Network
- Decision Tree
- Random Forest
- Gradient Boosting

## Outputs Generated

After running the pipeline, the following are created automatically:

```text
images/
  correlation_matrix.png
  target_correlation.png
  model_comparison_adjusted_r2.png
  train_test_r2.png
  train_test_adjusted_r2.png
  train_test_mse.png
  feature_importance.png
  kmeans_elbow.png
  kmeans_silhouette.png

reports/
  model_results.csv

models/
  best_model.pkl
  model_metadata.json
```

## Run Locally

```bash
pip install -r requirements.txt
python steel_clients_pipeline.py --data data/STEELMANUF_CLIENTS_SV.xlsx --sheet DB
python app.py
```

## Hugging Face Deployment

1. Create a new Hugging Face Space.
2. Select **Gradio** as the SDK.
3. Upload these files:

```text
app.py
steel_clients_pipeline.py
requirements.txt
data/STEELMANUF_CLIENTS_SV.xlsx
```

4. Hugging Face will install dependencies and run `app.py`.

## Tech Stack

Python, Pandas, NumPy, Scikit-Learn, Matplotlib, Gradio, Joblib
