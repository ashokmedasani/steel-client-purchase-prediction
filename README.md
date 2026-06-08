# Steel Client Purchase Prediction using Machine Learning

## Project Overview

This project analyzes client behavior for an American steel manufacturer, Alpha Steel, using the STEEL CLIENTS dataset. The dataset captures client interactions with a newly implemented web-based order-to-purchase system.

The goal of this project is to understand customer behavior, identify key factors influencing purchasing activity, and build predictive models to support digital sales strategy and client conversion.

## Business Problem

Alpha Steel introduced a web-based platform to automate the order-to-purchase process. The business objective is to evaluate how effectively the platform converts clients into buyers and reduces dependency on human client service agents.

## Dataset

The dataset includes client-level behavioral and transactional features such as:

* Sessions per year
* Average actions
* Number of purchases
* Pages visited
* Tons on cart
* Tons confirmed
* Delivery or pickup preference
* Web app sessions
* Executive-assisted sessions
* Excel tool usage
* Client catalogue usage
* Number of negotiations
* Client classification
* Construction and manufacturing indices

## Methods Used

* Data cleaning and validation
* Exploratory Data Analysis
* Outlier and anomaly handling
* Feature engineering
* Regression modeling
* Model comparison
* Cluster analysis using K-Means
* Performance evaluation using RMSE, R², and Adjusted R²

## Machine Learning Models

The following models were implemented and compared:

* OLS Regression
* Ridge Regression
* Lasso Regression
* Polynomial Regression
* Generalized Additive Model
* Neural Network
* Decision Tree
* Random Forest
* Gradient Boosting

## Key Result

Gradient Boosting achieved the strongest performance based on Test Adjusted R², followed by Random Forest. Non-linear models performed better than simple linear models, showing that client purchasing behavior contains non-linear patterns.

## Tech Stack

Python
Pandas
NumPy
Scikit-learn
PyGAM
TensorFlow / Keras
Matplotlib
Seaborn
Jupyter Notebook

## Business Value

This project helps the company:

* Understand client behavior on the web-based platform
* Identify high-value purchasing patterns
* Improve digital sales conversion
* Reduce dependency on manual client support
* Support data-driven decision-making in steel sales operations

## Future Improvements

* Deploy the best model using Hugging Face Spaces
* Add SHAP explainability
* Build an interactive prediction dashboard
* Add hyperparameter tuning
* Improve cluster interpretation for client segmentation
