import matplotlib
matplotlib.use("Agg")  # Required for Ubuntu/headless VMs

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, mean_absolute_percentage_error
)

import ruptures as rpt  # Bai–Perron-style structural breaks


# ============================================================
# 0. GLOBAL SETTINGS (Publication-ready)
# ============================================================
plt.style.use("seaborn-v0_8")
sns.set_context("talk")

plt.rcParams.update({
    "figure.figsize": (8, 4.5),
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.autolayout": True
})

BASE_DIR = Path(".")
DATA_PATH = BASE_DIR / "merged_bulgaria_monthly.xlsx"
FIG_DIR = BASE_DIR / "figures"
TABLE_DIR = BASE_DIR / "tables"

FIG_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)


# ============================================================
# 1. Johansen Test with Reimers Correction
# ============================================================
def run_johansen_corrected(data, lags=2):
    T, n = data.shape
    res = coint_johansen(data, det_order=0, k_ar_diff=lags)
    correction = (T - n * (lags + 1)) / T
    corrected_trace = res.lr1 * correction

    return pd.DataFrame({
        'Null hypothesis': [f'Rank ≤ {i}' for i in range(n)],
        'Trace statistic (corrected)': corrected_trace,
        'Critical value 95%': res.cvt[:, 1],
        'Significant (Trace > 95%)': corrected_trace > res.cvt[:, 1]
    })


# ============================================================
# 2. Metrics
# ============================================================
def get_metrics(y_true, y_pred):
    return {
        'R²': r2_score(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE (%)': mean_absolute_percentage_error(y_true, y_pred) * 100,
        'Bias': np.mean(y_pred - y_true)
    }


# ============================================================
# 3. Structural Break Detection
# ============================================================
def detect_structural_breaks(series, n_bkps=3):
    algo = rpt.Binseg(model="l2").fit(series.values)
    bkps = algo.predict(n_bkps=n_bkps)
    return bkps[:-1]


def save_pub_figure(fig, filename):
    fig.savefig(FIG_DIR / f"{filename}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{filename}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 4. Rolling Forecasts
# ============================================================
def rolling_forecast(model, X, y, initial_train_size, step=1):
    preds, trues, idx = [], [], []

    for start in range(initial_train_size, len(X), step):
        end = start + step
        if end > len(X):
            break

        model.fit(X.iloc[:start], y.iloc[:start])
        y_pred = model.predict(X.iloc[start:end])

        preds.extend(y_pred)
        trues.extend(y.iloc[start:end].values)
        idx.extend(X.iloc[start:end].index)

    return (
        pd.Series(trues, index=idx, name="Actual"),
        pd.Series(preds, index=idx, name="Rolling forecast")
    )


# ============================================================
# 5. Load Data
# ============================================================
df = pd.read_excel(DATA_PATH)

df['observation_date'] = pd.to_datetime(df['observation_date'], errors='coerce')

df = df.sort_values('observation_date').set_index('observation_date')

df = df[df['Country'] == 'Bulgaria']
df = df.dropna(subset=['HICP_Bulgaria', 'Brent_Oil_Price_USD'])


# ============================================================
# 6. VECX: Cointegration + Error Correction Term
# ============================================================
endog = df[['HICP_Bulgaria', 'Brent_Oil_Price_USD']]
exog = df[['Unemployment_Pct', 'Gross_Savings_Pct_GDP', 'Trade_Openness_Pct_GDP']]

johansen_table = run_johansen_corrected(endog)

vecm = VECM(endog, exog=exog, k_ar_diff=2, coint_rank=1, deterministic="n")
vecm_res = vecm.fit()

beta = vecm_res.beta[:, 0]
df['ECT'] = (endog.values @ beta)
df['ECT_lag1'] = df['ECT'].shift(1)


# ============================================================
# 7. ARDL Lags
# ============================================================
for lag in range(1, 4):
    df[f'HICP_lag{lag}'] = df['HICP_Bulgaria'].shift(lag)
    df[f'Brent_lag{lag}'] = df['Brent_Oil_Price_USD'].shift(lag)

df = df.dropna()

features = [c for c in df.columns if 'lag' in c] + [
    'Unemployment_Pct', 'Gross_Savings_Pct_GDP',
    'Trade_Openness_Pct_GDP', 'ECT_lag1'
]

X = df[features]
y = df['HICP_Bulgaria']
dates = df.index


# ============================================================
# 8. Train/Test Split
# ============================================================
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
dates_test = dates[split_idx:]


# ============================================================
# 9. Hybrid Stacking Model
# ============================================================
estimators = [
    ('ARDL', Ridge(alpha=1.0)),
    ('FAVAR_RF', RandomForestRegressor(
        n_estimators=300, max_depth=6, random_state=42
    ))
]

model = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(alpha=1.0)
)

model.fit(X_train, y_train)
pred_test = model.predict(X_test)

metrics_static = get_metrics(y_test, pred_test)


# ============================================================
# 10. Rolling Forecasts
# ============================================================
y_roll_true, y_roll_pred = rolling_forecast(
    model, X, y, initial_train_size=split_idx
)

metrics_roll = get_metrics(y_roll_true, y_roll_pred)


# ============================================================
# 11. Structural Breaks
# ============================================================
residuals = y_test - pred_test
bkps = detect_structural_breaks(residuals)


# ============================================================
# 12. Publication-Ready Figures
# ============================================================
# Figure 1: Actual vs Predicted
fig, ax = plt.subplots()
ax.plot(dates_test, y_test, label="Actual", color="black")
ax.plot(dates_test, pred_test, label="Predicted", color="blue")
ax.set_title("Actual vs Predicted HICP – Test Sample")
ax.set_xlabel("Time")
ax.set_ylabel("HICP (index)")
ax.legend(frameon=False)
save_pub_figure(fig, "Figure1_actual_vs_predicted")

# Figure 2: Rolling Forecasts
fig, ax = plt.subplots()
ax.plot(y.index, y, label="Actual", color="black")
ax.plot(y_roll_pred.index, y_roll_pred, label="Rolling forecast", color="orange")
ax.set_title("Rolling Out-of-Sample Forecasts – HICP")
ax.set_xlabel("Time")
ax.set_ylabel("HICP (index)")
ax.legend(frameon=False)
save_pub_figure(fig, "Figure2_rolling_forecasts")

# Figure 3: Structural Breaks
fig, ax = plt.subplots()
ax.plot(dates_test, residuals, label="Residuals", color="red")
for b in bkps:
    ax.axvline(dates_test[b], color="blue", linestyle="--")
ax.set_title("Residuals with Structural Breaks")
ax.set_xlabel("Time")
ax.set_ylabel("Residuals")
ax.legend(frameon=False)
save_pub_figure(fig, "Figure3_structural_breaks")


# ============================================================
# 13. Publication-Ready Tables (Excel + LaTeX)
# ============================================================
# Excel export
with pd.ExcelWriter("bulgaria_hybrid_results.xlsx") as writer:
    df.to_excel(writer, sheet_name="Data")
    pd.DataFrame(metrics_static, index=["Static"]).to_excel(writer, sheet_name="Metrics_Static")
    pd.DataFrame(metrics_roll, index=["Rolling"]).to_excel(writer, sheet_name="Metrics_Rolling")
    johansen_table.to_excel(writer, sheet_name="Johansen")

# LaTeX export
def save_latex_table(df, filename, caption, label):
    with open(TABLE_DIR / filename, "w") as f:
        f.write(df.round(3).to_latex(
            index=True,
            caption=caption,
            label=label,
            column_format="lrrrrr",
            escape=False
        ))

save_latex_table(pd.DataFrame(metrics_static, index=["Static"]),
                 "Table1_static_metrics.tex",
                 "Table 1. Static test performance metrics.",
                 "tab:static")

save_latex_table(pd.DataFrame(metrics_roll, index=["Rolling"]),
                 "Table2_rolling_metrics.tex",
                 "Table 2. Rolling forecast performance metrics.",
                 "tab:rolling")

save_latex_table(johansen_table,
                 "Table3_johansen.tex",
                 "Table 3. Johansen cointegration test with Reimers correction.",
                 "tab:johansen")

print("\nPipeline completed successfully.")
print("Figures saved in ./figures/")
print("Tables saved in ./tables/")
