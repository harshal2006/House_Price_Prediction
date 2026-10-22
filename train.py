"""
train.py — House Price Prediction Model (Training Module)
=========================================================
This module handles:
  1. Loading or generating the dataset (NO manual download required).
  2. Exploratory preview (first 5 rows, basic info).
  3. Training a Simple Linear Regression model.
  4. Evaluating the model with MAE, MSE, RMSE, and R² score.

It can be run standalone (`python train.py`) for a quick terminal demo,
or imported by the Streamlit app (`app.py`).
"""

import io
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1.  Dataset Loading — Automatic, No Manual Download
# ---------------------------------------------------------------------------

# A small, publicly accessible CSV hosted on GitHub (raw link).
DATASET_URL = (
    "https://raw.githubusercontent.com/dsrscientist/dataset1/master/census.csv"
)


def _try_load_real_dataset() -> pd.DataFrame | None:
    """
    Attempt to fetch a real-world housing dataset from a public GitHub URL.
    Returns a clean two-column DataFrame ['Area (sq ft)', 'Price (₹)'] or None.
    """
    try:
        raw = pd.read_csv(DATASET_URL, timeout=10)

        # The census.csv has columns like 'total_rooms', 'total_bedrooms',
        # 'median_house_value', etc.  We'll derive a usable Area-Price pair:
        #   • Area  ≈ total_rooms × 120  (approximate sq ft per room)
        #   • Price ≈ median_house_value converted to ₹ (×83 rough USD→INR)
        if "total_rooms" in raw.columns and "median_house_value" in raw.columns:
            df = pd.DataFrame()
            df["Area (sq ft)"] = (raw["total_rooms"] * 120).astype(int)
            df["Price (₹)"] = (raw["median_house_value"] * 83).astype(int)
            # Keep a reasonable subset (500 rows) for faster training & clear plots
            df = df.sample(n=min(500, len(df)), random_state=42).reset_index(drop=True)
            # Drop obvious outliers
            df = df[(df["Area (sq ft)"] > 200) & (df["Area (sq ft)"] < 10000)]
            df = df[(df["Price (₹)"] > 500_000) & (df["Price (₹)"] < 5_00_00_000)]
            df = df.reset_index(drop=True)
            return df
    except Exception:
        pass  # Network error, URL changed, etc. → fall back to synthetic data.
    return None


def _generate_synthetic_dataset(n: int = 200) -> pd.DataFrame:
    """
    Generate a small, realistic synthetic dataset using NumPy.
    Relationship:  Price ≈ 4500 × Area + noise  (in ₹)
    """
    rng = np.random.RandomState(42)
    area = rng.randint(500, 5000, size=n)
    price = (area * 4500 + rng.normal(0, 300_000, size=n)).astype(int)
    price = np.clip(price, 500_000, None)  # Minimum realistic price
    return pd.DataFrame({"Area (sq ft)": area, "Price (₹)": price})


def load_dataset() -> tuple[pd.DataFrame, str]:
    """
    Load the dataset — tries real data first, falls back to synthetic.

    Returns
    -------
    df : pd.DataFrame
        Two columns: 'Area (sq ft)', 'Price (₹)'.
    source : str
        'real' or 'synthetic' — so the UI can display an honest label.
    """
    df = _try_load_real_dataset()
    if df is not None and len(df) >= 30:
        return df, "real"
    return _generate_synthetic_dataset(), "synthetic"


# ---------------------------------------------------------------------------
# 2.  Dataset Preview Helpers
# ---------------------------------------------------------------------------

def dataset_info_string(df: pd.DataFrame) -> str:
    """Return the output of df.info() as a string (for display)."""
    buf = io.StringIO()
    df.info(buf=buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 3.  Model Training
# ---------------------------------------------------------------------------

def train_model(df: pd.DataFrame):
    """
    Train a Simple Linear Regression model.

    Returns
    -------
    model      : trained LinearRegression instance
    X_train, X_test, y_train, y_test : split arrays
    y_pred     : predictions on the test set
    metrics    : dict with MAE, MSE, RMSE, R²
    """
    X = df[["Area (sq ft)"]].values
    y = df["Price (₹)"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "MAE (₹)": f"{mae:,.0f}",
        "MSE": f"{mse:,.0f}",
        "RMSE (₹)": f"{rmse:,.0f}",
        "R² Score": f"{r2:.4f}",
    }

    return model, X_train, X_test, y_train, y_test, y_pred, metrics


def predict_price(model, area_sqft: float) -> float:
    """Predict the price for a given area in square feet."""
    return model.predict(np.array([[area_sqft]]))[0]


# ---------------------------------------------------------------------------
# 4.  Standalone Terminal Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  House Price Prediction — Training Script")
    print("=" * 60)

    # Load data
    df, source = load_dataset()
    print(f"\n📊 Dataset source: {source.upper()}")
    print(f"   Rows: {len(df)}  |  Columns: {list(df.columns)}\n")

    # Preview
    print("── First 5 rows ──")
    print(df.head().to_string(index=False))
    print()

    # Info
    print("── Dataset Info ──")
    print(dataset_info_string(df))

    # Train
    model, X_train, X_test, y_train, y_test, y_pred, metrics = train_model(df)

    print("── Model Evaluation ──")
    for k, v in metrics.items():
        print(f"   {k:12s} : {v}")

    # Interactive prediction
    print("\n── Predict a Price ──")
    try:
        area = float(input("Enter house area in sq ft: "))
        price = predict_price(model, area)
        print(f"   Estimated Price: ₹{price:,.0f}")
    except (ValueError, EOFError):
        print("   (skipped — no input)")

    print("\n✅ Done. Run `streamlit run app.py` for the web interface.")
