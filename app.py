import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Steel Clients Analytics",
    page_icon="📊",
    layout="wide"
)

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
]


def adjusted_r2_score(r2, n, p):
    if n <= p + 1:
        return np.nan
    return 1 - ((1 - r2) * (n - 1) / (n - p - 1))


@st.cache_data
def load_dataset(uploaded_file, local_path, sheet_name):
    if uploaded_file is not None:
        return pd.read_excel(uploaded_file, sheet_name=sheet_name)

    path = Path(local_path)
    if path.exists():
        return pd.read_excel(path, sheet_name=sheet_name)

    return None


def clean_dataset(df):
    df = df.copy()

    if "ID_CLIENT" in df.columns:
        df = df.drop(columns=["ID_CLIENT"])

    required = FEATURE_COLUMNS + [TARGET]
    missing = [col for col in required if col not in df.columns]

    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    model_df = df[required].copy()

    for col in model_df.columns:
        if col != "CLASSIFICATION":
            model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

    model_df["CLASSIFICATION"] = model_df["CLASSIFICATION"].astype(str)

    numeric_cols = [col for col in model_df.columns if col != "CLASSIFICATION"]
    model_df[numeric_cols] = model_df[numeric_cols].fillna(model_df[numeric_cols].median())

    return model_df


def prepare_features(model_df):
    X = model_df[FEATURE_COLUMNS].copy()
    y = model_df[TARGET].copy()

    X_encoded = pd.get_dummies(X, columns=["CLASSIFICATION"], drop_first=False)

    return X, X_encoded, y


def build_models():
    return {
        "OLS Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=0.25),
        "Lasso Regression": Lasso(alpha=0.25),
        "Polynomial Regression": Pipeline(
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


@st.cache_data
def train_models(X_encoded, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=0.20,
        random_state=42,
    )

    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_encoded.columns,
        index=X_train.index,
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_encoded.columns,
        index=X_test.index,
    )

    rows = []

    for model_name, model in build_models().items():
        model.fit(X_train_scaled, y_train)

        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)

        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)

        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)

        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)

        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)

        feature_count = X_train_scaled.shape[1]

        if model_name == "Polynomial Regression":
            feature_count = model.named_steps["poly"].transform(X_train_scaled).shape[1]

        rows.append(
            {
                "Model": model_name,
                "Train_R2": train_r2,
                "Test_R2": test_r2,
                "Train_Adjusted_R2": adjusted_r2_score(train_r2, len(y_train), feature_count),
                "Test_Adjusted_R2": adjusted_r2_score(test_r2, len(y_test), feature_count),
                "Train_MSE": train_mse,
                "Test_MSE": test_mse,
                "Train_RMSE": train_rmse,
                "Test_RMSE": test_rmse,
                "Train_MAE": train_mae,
                "Test_MAE": test_mae,
            }
        )

    results_df = pd.DataFrame(rows).sort_values(
        by=["Test_Adjusted_R2", "Test_R2"],
        ascending=False,
    )

    return results_df


def plot_correlation_matrix(model_df):
    numeric_df = model_df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(corr, aspect="auto")
    fig.colorbar(im, ax=ax, label="Correlation")

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    ax.set_title("Correlation Matrix - Steel Clients Dataset")

    fig.tight_layout()
    return fig


def plot_target_correlation(model_df):
    numeric_df = model_df.select_dtypes(include=[np.number])

    corr = (
        numeric_df.corr()[TARGET]
        .drop(TARGET)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(corr.index, corr.values)
    ax.set_title("Feature Correlation with Number of Purchases")
    ax.set_xlabel("Correlation")
    fig.tight_layout()

    return fig


def plot_model_bar(results_df, metric_col, title, xlabel):
    ordered = results_df.sort_values(metric_col)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(ordered["Model"], ordered[metric_col])
    ax.set_title(title)
    ax.set_xlabel(xlabel)

    for index, value in enumerate(ordered[metric_col]):
        ax.text(value, index, f" {value:.3f}", va="center", fontsize=8)

    fig.tight_layout()
    return fig


def plot_train_test_line(results_df, train_col, test_col, title, ylabel):
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        results_df["Model"],
        results_df[train_col],
        marker="o",
        linewidth=2,
        label=f"Train {ylabel}",
    )

    ax.plot(
        results_df["Model"],
        results_df[test_col],
        marker="s",
        linewidth=2,
        label=f"Test {ylabel}",
    )

    for x, y in zip(results_df["Model"], results_df[train_col]):
        ax.annotate(
            f"{y:.3f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )

    for x, y in zip(results_df["Model"], results_df[test_col]):
        ax.annotate(
            f"{y:.3f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, -14),
            ha="center",
            fontsize=8,
        )

    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    return fig


st.title("Steel Client Purchase Analytics using Machine Learning")

st.markdown(
    """
This app dynamically performs the full analytical workflow for the **STEEL CLIENTS** dataset.  
It reads the dataset, cleans the modeling fields, performs exploratory analysis, trains regression-based machine learning models, and compares model performance live.
"""
)

with st.sidebar:
    st.header("Dataset Settings")

    uploaded_file = st.file_uploader(
        "Upload Steel Clients Excel file",
        type=["xlsx"]
    )

    local_path = st.text_input(
        "Or use local dataset path",
        value="data/STEELMANUF_CLIENTS_SV.xlsx"
    )

    sheet_name = st.text_input(
        "Excel Sheet Name",
        value="DB"
    )

    run_analysis = st.button("Run Analysis")

if not run_analysis:
    st.info("Upload your dataset or keep the local path, then click **Run Analysis**.")
    st.stop()

df = load_dataset(uploaded_file, local_path, sheet_name)

if df is None:
    st.error("Dataset not found. Upload the Excel file or check the local path.")
    st.stop()

model_df = clean_dataset(df)
X_raw, X_encoded, y = prepare_features(model_df)
results_df = train_models(X_encoded, y)

best_model = results_df.iloc[0]

st.header("1. Business Understanding")

st.markdown(
    """
### Problem Statement

Alpha Steel introduced a web-based order-to-purchase platform to improve customer experience, automate purchase activity, and reduce dependency on manual client service agents.

The purpose of this analysis is to understand client purchasing behavior and identify which customer activities are most connected with `NUMBER_OF_PURCHASES`.

### Target Variable

`NUMBER_OF_PURCHASES`

### Business Questions

- Which customer behaviors are most related to purchase activity?
- Does web-platform engagement influence purchasing behavior?
- Which machine learning models best explain client purchase patterns?
"""
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{df.shape[0]:,}")
c2.metric("Columns", f"{df.shape[1]:,}")
c3.metric("Target", TARGET)
c4.metric("Modeling Features", len(FEATURE_COLUMNS))

with st.expander("View Dataset Preview"):
    st.dataframe(df.head(20), use_container_width=True)

with st.expander("View Selected Modeling Columns"):
    st.write(FEATURE_COLUMNS + [TARGET])
    st.dataframe(model_df.head(20), use_container_width=True)

st.header("2. Exploratory Analysis & Discoveries")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Correlation Matrix")
    st.pyplot(plot_correlation_matrix(model_df))

with col2:
    st.subheader("Feature Correlation with Purchases")
    st.pyplot(plot_target_correlation(model_df))

st.markdown(
    """
### Key Discoveries

- Customer engagement variables show meaningful relationships with purchase activity.
- Negotiation behavior is an important signal for understanding customer conversion.
- Web application usage supports analysis of digital platform adoption.
- Tons confirmed and purchase-related activity help explain client buying behavior.
"""
)

st.header("3. Machine Learning Model Evaluation")

st.markdown(
    """
The following regression-based machine learning models are trained and compared live:

- OLS Regression
- Ridge Regression
- Lasso Regression
- Polynomial Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Neural Network
"""
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Best Model", best_model["Model"])
m2.metric("Test R²", f"{best_model['Test_R2']:.3f}")
m3.metric("Test Adjusted R²", f"{best_model['Test_Adjusted_R2']:.3f}")
m4.metric("Test RMSE", f"{best_model['Test_RMSE']:.3f}")

st.subheader("Model Results Table")
st.dataframe(results_df, use_container_width=True)

st.subheader("Model Performance Proofs")

col3, col4 = st.columns(2)

with col3:
    st.pyplot(
        plot_model_bar(
            results_df,
            "Test_Adjusted_R2",
            "Model Comparison by Test Adjusted R²",
            "Test Adjusted R²"
        )
    )

with col4:
    st.pyplot(
        plot_model_bar(
            results_df,
            "Test_R2",
            "Model Comparison by Test R²",
            "Test R²"
        )
    )

st.pyplot(
    plot_train_test_line(
        results_df,
        "Train_R2",
        "Test_R2",
        "Train and Test R² for Different Models",
        "R²"
    )
)

st.pyplot(
    plot_train_test_line(
        results_df,
        "Train_Adjusted_R2",
        "Test_Adjusted_R2",
        "Train and Test Adjusted R² for Different Models",
        "Adjusted R²"
    )
)

st.pyplot(
    plot_train_test_line(
        results_df,
        "Train_MSE",
        "Test_MSE",
        "Train and Test MSE for Different Models",
        "MSE"
    )
)

st.header("4. Conclusions & Business Recommendations")

st.markdown(
    f"""
### Final Conclusion

The strongest performing model in this run is **{best_model["Model"]}**, based on test-side adjusted R² and R².

This confirms that customer purchasing behavior is better explained by models that can capture multiple behavioral signals and non-linear relationships.

### Business Recommendations

- Prioritize clients with strong digital engagement and repeated web activity.
- Track negotiation frequency as a potential indicator of sales conversion.
- Use tons-related activity as a behavioral signal for purchase intent.
- Encourage customers to use the web-based order system more actively.
- Apply machine learning insights to support customer segmentation and sales prioritization.

### Tools Used

Python | Pandas | NumPy | Scikit-learn | Matplotlib | Streamlit | Regression Analysis | Machine Learning
"""
)

st.caption("Project by Ashok Medasani | Steel Clients Predictive Analytics")
