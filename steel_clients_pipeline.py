"""
Steel Clients Predictive Analytics Pipeline
Author: Ashok Medasani

This script:
1. Loads the STEEL CLIENTS Excel file from data/
2. Cleans and prepares modeling columns
3. Trains regression models to predict NUMBER_OF_PURCHASES
4. Saves model comparison results into reports/model_results.csv
5. Saves the best model into models/best_model.pkl
6. Generates EDA and model performance images into images/

Run:
    python steel_clients_pipeline.py --data data/STEELMANUF_CLIENTS_SV.xlsx --sheet DB

Optional:
    python steel_clients_pipeline.py --data data/STEELMANUF_CLIENTS_SV.xlsx --sheet DB --no-images
"""

from __future__ import annotations

import argparse
import json
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
    for folder in ["images", "reports", "models"]:
        Path(folder).mkdir(parents=True, exist_ok=True)


def load_data(data_path: str, sheet_name: str = "DB") -> pd.DataFrame:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            "Place the Excel file inside data/ and pass the correct path."
        )

    df = pd.read_excel(path, sheet_name=sheet_name)

    if "ID_CLIENT" in df.columns:
        df = df.drop(columns=["ID_CLIENT"])

    return df


def prepare_model_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "These expected columns are missing from the dataset:\n"
            + "\n".join(missing)
        )

    model_df = df[FEATURE_COLUMNS].copy()

    # Fill missing values before modeling
    for col in model_df.columns:
        if model_df[col].dtype == "object":
            model_df[col] = model_df[col].fillna(model_df[col].mode()[0])
        else:
            model_df[col] = model_df[col].fillna(model_df[col].median())

    # Convert CLASSIFICATION into dummy variables
    model_df = pd.get_dummies(model_df, columns=["CLASSIFICATION"], drop_first=False)

    X = model_df.drop(columns=[TARGET])
    y = model_df[TARGET]

    return X, y


def adjusted_r2_score(r2: float, n: int, p: int) -> float:
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
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    return {
        "Model": model_name,
        "Train_R2": train_r2,
        "Test_R2": test_r2,
        "Train_Adj_R2": adjusted_r2_score(train_r2, len(y_train), n_features),
        "Test_Adj_R2": adjusted_r2_score(test_r2, len(y_test), n_features),
        "Train_MSE": train_mse,
        "Test_MSE": test_mse,
        "Train_RMSE": np.sqrt(train_mse),
        "Test_RMSE": np.sqrt(test_mse),
    }


def build_models() -> Dict[str, object]:
    return {
        "OLS Regression": LinearRegression(),
        "Ridge Regression (alpha=0.25)": Ridge(alpha=0.25),
        "Lasso Regression (alpha=0.25)": Lasso(alpha=0.25),
        "Polynomial Regression (Degree=2)": Pipeline(
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
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
        if name == "Polynomial Regression (Degree=2)":
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
        by=["Test_Adj_R2", "Test_R2"],
        ascending=False,
    )

    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

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

    with open("models/model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "target": TARGET,
                "best_model_name": best_model_name,
                "feature_columns": list(X.columns),
            },
            f,
            indent=4,
        )

    full_scaled_X = pd.DataFrame(
        scaler.transform(X),
        columns=X.columns,
        index=X.index,
    )

    return results_df, best_model, scaler, full_scaled_X


def save_correlation_heatmap(df: pd.DataFrame) -> None:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return

    corr = numeric_df.corr()

    plt.figure(figsize=(16, 11))
    plt.imshow(corr, aspect="auto")
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=7)
    plt.title("Correlation Matrix - Steel Clients Dataset", fontsize=14)
    plt.tight_layout()
    plt.savefig("images/correlation_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_target_correlation_bar(df: pd.DataFrame) -> None:
    numeric_df = df.select_dtypes(include=[np.number])
    if TARGET not in numeric_df.columns:
        return

    corr = numeric_df.corr()[TARGET].drop(TARGET).sort_values(ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(corr.index, corr.values)
    plt.title("Feature Correlation with Number of Purchases")
    plt.xlabel("Correlation")
    plt.tight_layout()
    plt.savefig("images/target_correlation.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_model_comparison_chart(results_df: pd.DataFrame) -> None:
    ordered = results_df.sort_values("Test_Adj_R2", ascending=True)

    plt.figure(figsize=(12, 7))
    plt.barh(ordered["Model"], ordered["Test_Adj_R2"])
    plt.title("Model Comparison by Test Adjusted R²", fontsize=16)
    plt.xlabel("Test Adjusted R²")
    plt.tight_layout()
    plt.savefig("images/model_comparison_adjusted_r2.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_train_test_metrics(results_df: pd.DataFrame) -> None:
    """
    Saves three line charts:
    1. Train vs Test R²
    2. Train vs Test Adjusted R²
    3. Train vs Test MSE
    """
    metric_plots = [
        ("Train_R2", "Test_R2", "R²", "train_test_r2.png"),
        ("Train_Adj_R2", "Test_Adj_R2", "Adjusted R²", "train_test_adjusted_r2.png"),
        ("Train_MSE", "Test_MSE", "Mean Squared Error (MSE)", "train_test_mse.png"),
    ]

    plot_df = results_df.copy()

    # Use readable order from simple to complex
    model_order = [
        "OLS Regression",
        "Ridge Regression (alpha=0.25)",
        "Lasso Regression (alpha=0.25)",
        "Polynomial Regression (Degree=2)",
        "Neural Network",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
    ]
    plot_df["Model"] = pd.Categorical(plot_df["Model"], categories=model_order, ordered=True)
    plot_df = plot_df.sort_values("Model")

    for train_col, test_col, ylabel, filename in metric_plots:
        plt.figure(figsize=(14, 6))

        plt.plot(
            plot_df["Model"].astype(str),
            plot_df[train_col],
            marker="o",
            linewidth=2,
            label=f"Train {ylabel}",
        )

        plt.plot(
            plot_df["Model"].astype(str),
            plot_df[test_col],
            marker="s",
            linewidth=2,
            label=f"Test {ylabel}",
        )

        for x, y in zip(plot_df["Model"].astype(str), plot_df[train_col]):
            plt.annotate(
                f"{y:.3f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
            )

        for x, y in zip(plot_df["Model"].astype(str), plot_df[test_col]):
            plt.annotate(
                f"{y:.3f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, -15),
                ha="center",
                fontsize=8,
            )

        plt.title(f"Train and Test {ylabel} for Different Models", fontsize=16)
        plt.xlabel("Model")
        plt.ylabel(ylabel)
        plt.xticks(rotation=45, ha="right")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.savefig(f"images/{filename}", dpi=300, bbox_inches="tight")
        plt.close()


def save_feature_importance(best_model: object, feature_names: list[str]) -> None:
    """
    Saves feature importance if the best model supports feature_importances_.
    """
    if not hasattr(best_model, "feature_importances_"):
        return

    importances = pd.Series(best_model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True).tail(15)

    plt.figure(figsize=(10, 7))
    plt.barh(importances.index, importances.values)
    plt.title("Top Feature Importances from Best Model")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("images/feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_cluster_analysis_chart(X_scaled: pd.DataFrame) -> None:
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
    plt.savefig("images/kmeans_elbow.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(list(cluster_range), silhouette_scores, marker="o")
    plt.title("KMeans Silhouette Scores")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.tight_layout()
    plt.savefig("images/kmeans_silhouette.png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_images(
    df: pd.DataFrame,
    results_df: pd.DataFrame,
    X_scaled: pd.DataFrame,
    best_model: object,
    feature_names: list[str],
) -> None:
    save_correlation_heatmap(df)
    save_target_correlation_bar(df)
    save_model_comparison_chart(results_df)
    plot_train_test_metrics(results_df)
    save_feature_importance(best_model, feature_names)
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
        generate_images(df, results_df, X_scaled, best_model, list(X.columns))

    print("\nDone.")
    print("Saved outputs:")
    print("- reports/model_results.csv")
    print("- models/best_model.pkl")
    print("- models/model_metadata.json")
    if not args.no_images:
        print("- images/correlation_matrix.png")
        print("- images/target_correlation.png")
        print("- images/model_comparison_adjusted_r2.png")
        print("- images/train_test_r2.png")
        print("- images/train_test_adjusted_r2.png")
        print("- images/train_test_mse.png")
        print("- images/feature_importance.png if supported by best model")
        print("- images/kmeans_elbow.png")
        print("- images/kmeans_silhouette.png")


if __name__ == "__main__":
    main()
