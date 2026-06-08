# Steel Client Purchase Analytics using Machine Learning

## Overview

This project analyzes customer purchasing behavior for **Alpha Steel** using a machine learning-driven analytical framework. The study focuses on understanding how client engagement, digital platform activity, negotiations, and purchasing patterns influence customer buying behavior within a web-based order-to-purchase ecosystem.

The objective was not only to identify the key drivers of purchases but also to evaluate multiple regression-based machine learning models and determine which approach best captures customer behavior patterns.

---

## Business Problem

Alpha Steel implemented a digital order-to-purchase platform to streamline customer interactions and reduce dependency on manual sales processes.

With increasing customer activity occurring through digital channels, the company needed answers to the following questions:

* Which customer behaviors are most strongly associated with purchases?
* Does digital platform engagement influence buying decisions?
* Which client attributes contribute most to purchasing activity?
* Which machine learning model best explains customer purchase patterns?

The ultimate goal was to leverage analytics to support data-driven decision-making and improve sales effectiveness.

---

## Dataset

The analysis was performed using the **Steel Clients Dataset**, which contains behavioral and transactional information collected from clients interacting with Alpha Steel's digital platform.

### Key Features

* Number of Negotiations
* Sessions per Year
* Sessions on Web Application
* Tons Confirmed
* Delivery or Pickup Preference
* Sessions Attended by Executive
* Average Actions
* Maximum Pages Visited
* Average Pages Visited
* Use of Excel Tool
* Use of Client Catalogue
* Customer Classification

### Target Variable

```text
NUMBER_OF_PURCHASES
```

---

## Project Workflow

### 1. Data Preparation

* Data cleaning and validation
* Missing value treatment
* Feature selection
* Categorical encoding
* Dataset standardization

### 2. Exploratory Data Analysis

The exploratory analysis focused on understanding relationships between customer activities and purchasing behavior.

Key activities included:

* Correlation analysis
* Behavioral pattern exploration
* Feature importance assessment
* Customer activity evaluation
* Business interpretation of analytical findings

For the complete exploratory analysis, please review the notebooks and reports included in this repository.

### 3. Machine Learning Modeling

Multiple regression-based machine learning models were trained and evaluated.

Models compared:

* OLS Regression
* Ridge Regression
* Lasso Regression
* Polynomial Regression
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor
* Neural Network Regressor

### 4. Model Evaluation

Models were evaluated using:

* R² Score
* Adjusted R² Score
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Error (MAE)

---

## Key Discoveries

The analysis revealed several meaningful business insights:

### Customer Engagement Matters

Customers demonstrating higher platform engagement generally exhibited stronger purchasing activity.

### Negotiation Activity is a Strong Indicator

The frequency of negotiations showed meaningful relationships with purchase behavior and conversion potential.

### Digital Platform Adoption Creates Valuable Signals

Web application activity provides measurable indicators that can support customer prioritization and sales planning.

### Purchase Behavior is Non-Linear

Traditional linear regression approaches were outperformed by ensemble-based machine learning methods, indicating that customer behavior contains complex non-linear relationships.

---

## Results

The project compared multiple machine learning approaches and ranked them based on predictive performance.

The strongest-performing models demonstrated significantly better ability to capture customer purchasing behavior compared to traditional regression methods.

Interactive model comparison charts, evaluation metrics, and visual proofs are available through the Streamlit application included in this repository.

---

## Streamlit Application

This repository includes a fully interactive Streamlit application that performs the complete analytical workflow dynamically.

### Features

* Dataset loading
* Data preprocessing
* Exploratory Data Analysis
* Correlation Analysis
* Feature Correlation Analysis
* Model Training
* Model Comparison
* Performance Evaluation
* Business Recommendations

The application generates all visualizations directly from the dataset and does not rely on static screenshots.

Run locally:

```bash
streamlit run app.py
```

---

## Repository Structure

```text
steel-client-purchase-analytics/
│
├── data/
│   └── STEELMANUF_CLIENTS_SV.xlsx
│
├── notebooks/
│   └── project_notebooks.ipynb
│
├── reports/
│   └── project_reports
│
├── app.py
├── steel_clients_pipeline.py
├── requirements.txt
└── README.md
```

---

## Technologies Used

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### Machine Learning

* Scikit-Learn

### Visualization

* Plotly
* Matplotlib

### Application Development

* Streamlit

---

## Business Value

This project demonstrates how machine learning and analytics can be applied to:

* Understand customer purchasing behavior
* Improve digital sales strategies
* Support customer segmentation initiatives
* Identify high-value client activity patterns
* Enable data-driven business decision-making

---

## Author

**Ashok Medasani**

Master's in Applied Analytics
Saint Louis University

Areas of Interest:

* Data Analytics
* Business Analytics
* Machine Learning
* Predictive Analytics
* Data Engineering
* Decision Intelligence
