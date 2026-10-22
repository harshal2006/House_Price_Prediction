# 🏠 House Price Prediction Using Linear Regression

A beginner-friendly **Predictive Analytics** project that predicts house prices based on the area of a house (in square feet) using **Simple Linear Regression**.

> **Subject:** Predictive Analytics  
> **Course:** BTech CSE  
> **Algorithm:** Simple Linear Regression (Supervised Learning)

---

## 📌 Project Objective

Build a machine learning model that learns the relationship between a house's area (sq ft) and its price (₹), then predicts the price of a new house given its area.

---

## 🧠 How Linear Regression Works (Simple Explanation)

Linear Regression finds the **best-fit straight line** through the data.

```
Price = m × Area + c
```

- **m** (slope) — how much the price increases for each extra square foot.
- **c** (intercept) — the base price when area is theoretically zero.

The algorithm adjusts `m` and `c` to **minimise the total error** between predicted and actual prices. This method is called **Ordinary Least Squares (OLS)**.

Once trained, the model uses this equation to predict the price for any new area value.

---

## 📊 Dataset

The project **does not require any manual dataset download**. It works automatically:

1. **Primary:** Fetches a real-world housing dataset from a public GitHub URL.
2. **Fallback:** If the URL is unreachable (no internet, URL changed), it generates a realistic **synthetic dataset** using NumPy.

The app clearly indicates whether it is using real or synthetic data.

---

## 🛠️ Technology Stack

| Tool | Purpose |
|------|---------|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations & synthetic data |
| Matplotlib | Data visualisation |
| Scikit-learn | Linear Regression model & metrics |
| Streamlit | Web interface |

---

## 📁 Project Structure

```
PA_project/
├── app.py              # Streamlit web application
├── train.py            # Model training & data loading module
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── .gitignore          # Git ignore rules
```

---

## 🚀 How to Run

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### macOS / Linux

```bash
# 1. Navigate to the project folder
cd PA_project

# 2. Create a virtual environment
python3 -m venv .venv

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the web application
streamlit run app.py
```

### Windows

```powershell
# 1. Navigate to the project folder
cd PA_project

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate the virtual environment
.venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the web application
streamlit run app.py
```

### Terminal-Only Mode (Optional)

```bash
python train.py
```

This runs the training script in the terminal, shows the first 5 rows, evaluation metrics, and lets you type a house area for a quick prediction.

---

## 📐 Evaluation Metrics

The model is evaluated on the **test set (20% of data)** using:

| Metric | What It Measures |
|--------|-----------------|
| **MAE** (Mean Absolute Error) | Average absolute difference between predicted and actual prices |
| **MSE** (Mean Squared Error) | Average of squared differences (penalises large errors) |
| **RMSE** (Root Mean Squared Error) | Square root of MSE — same unit as price |
| **R² Score** | Proportion of variance explained (1.0 = perfect fit) |

---

## 🖥️ Web Interface Features

- 📊 Dataset preview (first 5 rows)
- 🔮 Interactive slider to set house area (sq ft)
- 🏷️ Predicted price displayed in Indian Rupees (₹)
- 📈 Scatter plot with regression line
- 📈 Actual vs Predicted comparison chart
- 📐 All four evaluation metrics at a glance
- ⚠️ Disclaimer noting that predictions are estimates

---

## 📤 Upload to GitHub

```bash
# Initialise a Git repository
git init
git add .
git commit -m "Initial commit: House Price Prediction project"

# Add your GitHub repository as remote (replace URL with yours)
git remote add origin https://github.com/YOUR_USERNAME/house-price-prediction.git

# Push to GitHub
git branch -M main
git push -u origin main
```

> Replace `YOUR_USERNAME` with your actual GitHub username and create the repository on GitHub first.

---

## 🎓 Presentation Guide

When presenting this project, cover these points:

1. **Problem Statement** — Predicting house prices from area using ML.
2. **Algorithm** — Simple Linear Regression (supervised learning, OLS method).
3. **Dataset** — Auto-fetched from GitHub or generated synthetically.
4. **Steps** — Load → Split (80:20) → Train → Predict → Evaluate.
5. **Metrics** — Explain MAE, MSE, RMSE, R² and what your values mean.
6. **Demo** — Show the Streamlit app, enter an area, show the prediction.
7. **Limitations** — Single feature, linear assumption, dataset size.
8. **Future Scope** — Multiple features, polynomial regression, larger datasets.

---

## ⚠️ Disclaimer

Predictions are **estimates** based on a limited dataset and a simple linear model. Real house prices depend on many factors (location, rooms, age, market conditions). This project is for **educational purposes only**.

---

## 📝 License

This project is created for academic purposes. Feel free to use and modify.
