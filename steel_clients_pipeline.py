"""
Steel Clients Predictive Analytics Pipeline
Author: Ashok Medasani

What this script does:
1. Loads the STEEL CLIENTS Excel file from data/
2. Cleans and prepares selected modeling columns
3. Generates EDA/model images automatically into images/
4. Trains regression models to predict NUMBER_OF_PURCHASES
5. Saves model comparison results into reports/model_results.csv
6. Saves the best model into models/best_model.pkl

Run:
    python steel_clients_pipeline.py --data data/STEELMANUF_CLIENTS_SV.xlsx --sheet DB

Optional:
    python steel_clients_pipeline.py --data data/STEELMANUF_CLIENTS_SV.xlsx --sheet DB --no-images
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")


TARGET = "NUMBER_OF_PURCHASES"

FEATURE_COLUMNS = [
    "NUMBER_OF_NEGOTIATIONS",
    "SESSIONS_YEAR",
    "TONS_CONFIRMED",
    "SESSIONS_ONWEBAPP",
    "CHANGE_DISTCENTER",
    "DELIVERY_OR_PICKUP",
    "SESSIONS_ATTENDED_BYEXECUTIVE",
    "AVERAGE_ACTIONS",
    "MAXNUMBER_PAGES_VISITED",
    "AVGNUMBER_PAGES_VISITED",
    "USE_OF_EXCEL_TOOL",
    "USE_OF_CLIENT_CATALOGUE",
    "CLASSIFICATION",
    TARGET,
]


def ensure_dirs() -> None:
    """Create output folders if they do not already exist."""
    for folder in ["images", "reports", "models"]:
        Path(folder).mkdir(parents=True, exist_ok=True)


def load_data(data_path: str, sheet_name: str = "DB") -> pd.DataFrame:
    """Load the Excel dataset."""
    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            "Place your Excel file inside the data/ folder and pass the correct path."
        )

    df = pd.read_excel(path, sheet_name=sheet_name)

    if "ID_CLIENT" in df.columns:
        df = df.drop(columns=["ID_CLIENT"])

    return df


def prepare_model_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Select project-specific features and encode CLASSIFICATION.

    Target:
        NUMBER_OF_PURCHASES
    """
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "These expected columns are missing from the dataset:\n"
            + "\n".join(missing)
        )

    model_df = df[FEATURE_COLUMNS].copy()

    # Convert classification into dummy variables
    model_df = pd.get_dummies(model_df, columns=["CLASSIFICATION"], drop_first=False)

    X = model_df.drop(columns=[TARGET])
    y = model_df[TARGET]

    return X, y


def adjusted_r2_score(r2: float, n: int, p: int) -> float:
    """Calculate adjusted R²."""
    if n <= p + 1:
        return np.nan
    return 1 - ((1 - r2) * (n - 1) / (n - p - 1))


def evaluate_predictions(
    model_name: str,
    y_train: pd.Series,
    y_train_pred: np.ndarray,
    y_test: pd.Series,
    y_test_pred: np.ndarray,
    n_features: int,
) -> Dict[str, float]:
    """Return regression metrics for train and test predictions."""
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    return {
        "Model": model_name,
        "Train_RMSE": train_rmse,
        "Test_RMSE": test_rmse,
        "Train_R2": train_r2,
        "Test_R2": test_r2,
        "Train_Adjusted_R2": adjusted_r2_score(train_r2, len(y_train), n_features),
        "Test_Adjusted_R2": adjusted_r2_score(test_r2, len(y_test), n_features),
    }


def build_models() -> Dict[str, object]:
    """Define models used in the original project."""
    return {
        "OLS Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=0.25),
        "Lasso Regression": Lasso(alpha=0.25),
        "Polynomial Regression Degree 2": Pipeline(
            steps=[
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("linear", LinearRegression()),
            ]
        ),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
        ),
        "Neural Network": MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=1000,
            random_state=42,
        ),
    }


def train_models(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, object, StandardScaler, pd.DataFrame]:
    """
    Split data, scale features, train models, and return comparison table.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    results = []
    trained_models = {}

    for name, model in build_models().items():
        model.fit(X_train_scaled, y_train)

        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)

        n_features = X_train_scaled.shape[1]
        if name == "Polynomial Regression Degree 2":
            # Approximate adjusted R² using transformed feature count
            n_features = model.named_steps["poly"].transform(X_train_scaled).shape[1]

        results.append(
            evaluate_predictions(
                name,
                y_train,
                y_train_pred,
                y_test,
                y_test_pred,
                n_features,
            )
        )
        trained_models[name] = model

    results_df = pd.DataFrame(results).sort_values(
        by=["Test_Adjusted_R2", "Test_R2"],
        ascending=False,
    )

    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    # Save comparison table and best model
    results_df.to_csv("reports/model_results.csv", index=False)
    joblib.dump(
        {
            "model": best_model,
            "scaler": scaler,
            "feature_columns": list(X.columns),
            "target": TARGET,
            "best_model_name": best_model_name,
        },
        "models/best_model.pkl",
    )

    full_scaled_X = pd.DataFrame(
        scaler.transform(X),
        columns=X.columns,
        index=X.index,
    )

    return results_df, best_model, scaler, full_scaled_X


def save_correlation_heatmap(df: pd.DataFrame) -> None:
    """Save a correlation heatmap for numeric variables."""
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        return

    corr = numeric_df.corr()

    plt.figure(figsize=(14, 10))
    plt.imshow(corr, aspect="auto")
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=7)
    plt.title("Correlation Matrix - Steel Clients Dataset")
    plt.tight_layout()
    plt.savefig("images/correlation_matrix.png", dpi=300)
    plt.close()


def save_target_correlation_bar(df: pd.DataFrame) -> None:
    """Save correlation of numeric features with NUMBER_OF_PURCHASES."""
    numeric_df = df.select_dtypes(include=[np.number])

    if TARGET not in numeric_df.columns:
        return

    corr = (
        numeric_df.corr()[TARGET]
        .drop(TARGET)
        .sort_values(ascending=True)
    )

    plt.figure(figsize=(10, 8))
    plt.barh(corr.index, corr.values)
    plt.title("Feature Correlation with Number of Purchases")
    plt.xlabel("Correlation")
    plt.tight_layout()
    plt.savefig("images/target_correlation.png", dpi=300)
    plt.close()


def save_model_comparison_chart(results_df: pd.DataFrame) -> None:
    """Save model comparison chart based on Test Adjusted R²."""
    ordered = results_df.sort_values("Test_Adjusted_R2", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(ordered["Model"], ordered["Test_Adjusted_R2"])
    plt.title("Model Comparison by Test Adjusted R²")
    plt.xlabel("Test Adjusted R²")
    plt.tight_layout()
    plt.savefig("images/model_comparison_adjusted_r2.png", dpi=300)
    plt.close()


def save_cluster_analysis_chart(X_scaled: pd.DataFrame) -> None:
    """Save elbow and silhouette score chart for KMeans clustering."""
    cluster_range = range(2, 11)
    wcss = []
    silhouette_scores = []

    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels))

    plt.figure(figsize=(8, 5))
    plt.plot(list(cluster_range), wcss, marker="o")
    plt.title("KMeans Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.tight_layout()
    plt.savefig("images/kmeans_elbow.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(list(cluster_range), silhouette_scores, marker="o")
    plt.title("KMeans Silhouette Scores")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.tight_layout()
    plt.savefig("images/kmeans_silhouette.png", dpi=300)
    plt.close()


def generate_images(df: pd.DataFrame, results_df: pd.DataFrame, X_scaled: pd.DataFrame) -> None:
    """Generate all charts used for GitHub README/project explanation."""
    save_correlation_heatmap(df)
    save_target_correlation_bar(df)
    save_model_comparison_chart(results_df)
    save_cluster_analysis_chart(X_scaled)


def main() -> None:
    parser = argparse.ArgumentParser(description="Steel Clients ML Pipeline")
    parser.add_argument(
        "--data",
        default="data/STEELMANUF_CLIENTS_SV.xlsx",
        help="Path to Excel dataset.",
    )
    parser.add_argument(
        "--sheet",
        default="DB",
        help="Excel sheet name.",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip image generation.",
    )

    args = parser.parse_args()

    ensure_dirs()

    print("Loading data...")
    df = load_data(args.data, args.sheet)

    print("Preparing model data...")
    X, y = prepare_model_data(df)

    print("Training models...")
    results_df, best_model, scaler, X_scaled = train_models(X, y)

    print("\nModel Results:")
    print(results_df.to_string(index=False))

    if not args.no_images:
        print("\nGenerating images...")
        generate_images(df, results_df, X_scaled)

    print("\nDone.")
    print("Saved outputs:")
    print("- reports/model_results.csv")
    print("- models/best_model.pkl")
    if not args.no_images:
        print("- images/correlation_matrix.png")
        print("- images/target_correlation.png")
        print("- images/model_comparison_adjusted_r2.png")
        print("- images/kmeans_elbow.png")
        print("- images/kmeans_silhouette.png")


if __name__ == "__main__":
    main()
