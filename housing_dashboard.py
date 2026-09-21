import os
import warnings
import numpy as np
import pandas as pd
import panel as pn
import plotly.express as px
import plotly.graph_objects as go

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (
    ShuffleSplit,
    KFold,
    train_test_split,
    validation_curve,
    cross_val_score,
    GridSearchCV,
)
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

pn.extension("plotly", "tabulator", sizing_mode="stretch_width")


# ---------------------------------------------------------
# 1. DATA LOADING & PREPROCESSING
# ---------------------------------------------------------
@pn.cache
def load_data():
    if os.path.exists("housing.csv"):
        df = pd.read_csv("housing.csv")
    else:
        np.random.seed(42)
        n = 489
        df = pd.DataFrame(
            {
                "RM": np.random.normal(6.2, 0.7, n).clip(3.5, 9.0),
                "LSTAT": np.random.uniform(1.5, 38.0, n),
                "PTRATIO": np.random.uniform(12.0, 22.0, n),
                "MEDV": np.random.uniform(100000, 1000000, n),
            }
        )

    df["Price_Tier"] = pd.qcut(
        df["MEDV"], q=3, labels=["Affordable", "Mid-Range", "Luxury"]
    )

    df["Housing_Summary_Text"] = (
        "Property record with " + df["RM"].round(1).astype(str) + " rooms, "
        "LSTAT value of " + df["LSTAT"].round(1).astype(str) + "%, and a pupil-teacher ratio of "
        + df["PTRATIO"].round(1).astype(str) + ". Estimated value is $"
        + df["MEDV"].map("{:,.0f}".format) + " (" + df["Price_Tier"].astype(str) + " tier)."
    )

    return df


df = load_data()
numeric_cols = ["RM", "LSTAT", "PTRATIO", "MEDV"]
TIER_ORDER = ["Affordable", "Mid-Range", "Luxury"]


# ---------------------------------------------------------
# 2. CONTROLLERS & WIDGETS
# ---------------------------------------------------------
price_tier_select = pn.widgets.MultiSelect(
    name="Select Price Tiers",
    options=TIER_ORDER,
    value=TIER_ORDER,
)

rm_slider = pn.widgets.RangeSlider(
    name="Filter by Rooms (RM)",
    start=float(df["RM"].min()),
    end=float(df["RM"].max()),
    value=(float(df["RM"].min()), float(df["RM"].max())),
    step=0.1,
)

target_var_select = pn.widgets.Select(
    name="ML Target Variable", options=numeric_cols, value="MEDV"
)

k_folds_slider = pn.widgets.IntSlider(
    name="K-Fold Cross Validation Folds (K)", start=2, end=10, step=1, value=5
)

max_depth_slider = pn.widgets.IntSlider(
    name="Decision Tree Max Depth (Complexity)", start=1, end=15, step=1, value=10
)

# Interactive Grid Search Sliders
gs_max_depth_range = pn.widgets.IntRangeSlider(
    name="Grid Search Depth Range", start=1, end=15, value=(1, 10), step=1
)

gs_min_samples_split = pn.widgets.MultiChoice(
    name="Grid Search Min Samples Split",
    options=[2, 5, 10, 20],
    value=[2, 5, 10],
)


def filter_dataframe(tiers, rm_range):
    if not tiers:
        return df.iloc[0:0]
    filtered = df[df["Price_Tier"].isin(tiers)]
    filtered = filtered[(filtered["RM"] >= rm_range[0]) & (filtered["RM"] <= rm_range[1])]
    return filtered


def _empty_notice(title):
    """Standard placeholder figure/message when a filter leaves no rows."""
    fig = go.Figure()
    fig.update_layout(
        title=f"{title} — no data for current filters",
        annotations=[dict(text="No rows match the selected filters",
                           xref="paper", yref="paper", showarrow=False, font=dict(size=14))],
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------
# 3. STATISTICAL COMPONENT GENERATORS
# ---------------------------------------------------------
def get_table(tiers, rm_range):
    filtered = filter_dataframe(tiers, rm_range)
    if filtered.empty:
        return pn.pane.Markdown("**No data for current filters.**")
    stats = filtered[numeric_cols].describe().T[["count", "mean", "std", "min", "50%", "max"]]
    stats = stats.reset_index().rename(columns={"index": "Indicator", "50%": "median"})
    return pn.widgets.Tabulator(stats, pagination="remote", page_size=10, height=250)


def get_pie_chart(tiers, rm_range):
    filtered = filter_dataframe(tiers, rm_range)
    if filtered.empty:
        return _empty_notice("Price Tier Distribution")
    counts = filtered["Price_Tier"].value_counts().reset_index()
    fig = px.pie(
        counts, values="count", names="Price_Tier",
        title="Price Tier Distribution",
        hole=0.4, color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig


def get_bar_chart(tiers, rm_range, target_col="MEDV"):
    filtered = filter_dataframe(tiers, rm_range)
    if filtered.empty:
        return _empty_notice(f"Mean {target_col} by Tier")
    avg_df = filtered.groupby("Price_Tier", observed=True)[target_col].mean().reset_index()
    fig = px.bar(
        avg_df, x="Price_Tier", y=target_col,
        title=f"Mean {target_col} by Tier",
        color="Price_Tier", template="plotly_white"
    )
    return fig


def get_scatter_plot(tiers, rm_range, target_col="MEDV"):
    filtered = filter_dataframe(tiers, rm_range)
    if filtered.empty:
        return _empty_notice(f"LSTAT vs. {target_col}")
    # Pick an x-axis and a bubble-size feature that aren't the same as the target
    other_cols = [c for c in numeric_cols if c != target_col]
    x_col = "LSTAT" if target_col != "LSTAT" else other_cols[0]
    size_col = "RM" if target_col != "RM" and x_col != "RM" else other_cols[-1]
    fig = px.scatter(
        filtered, x=x_col, y=target_col,
        size=size_col, color="Price_Tier",
        title=f"{x_col} vs. {target_col} (Size = {size_col})",
        template="plotly_white"
    )
    return fig


def get_heatmap(tiers, rm_range):
    filtered = filter_dataframe(tiers, rm_range)
    if len(filtered) < 2:
        return _empty_notice("Feature Correlation Matrix")
    corr = filtered[numeric_cols].corr()
    fig = px.imshow(
        corr, text_auto=".2f",
        title="Feature Correlation Matrix",
        color_continuous_scale="RdBu_r"
    )
    return fig


# ---------------------------------------------------------
# 4. LINEAR & POLYNOMIAL (SQUARE) REGRESSION PREDICTION MODULE
# ---------------------------------------------------------
@pn.depends(target_var_select.param.value)
def run_linear_vs_polynomial_regression(target_col):
    ml_df = df[numeric_cols].dropna()
    X = ml_df.drop(columns=[target_col])
    y = ml_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_r2 = r2_score(y_test, lr_preds)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))

    # 2. Square / Polynomial Regression (Degree 2)
    poly = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    poly.fit(X_train, y_train)
    poly_preds = poly.predict(X_test)
    poly_r2 = r2_score(y_test, poly_preds)
    poly_rmse = np.sqrt(mean_squared_error(y_test, poly_preds))

    # Actual vs Predicted Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test, y=lr_preds, mode="markers", name="Linear Regression"))
    fig.add_trace(go.Scatter(x=y_test, y=poly_preds, mode="markers", name="Square (Poly Deg 2)"))
    fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
                             mode="lines", name="Ideal Fit", line=dict(dash="dash", color="black")))

    fig.update_layout(
        title=f"Regression Predictions vs Actual ({target_col})",
        xaxis_title=f"Actual {target_col}",
        yaxis_title=f"Predicted {target_col}",
        template="plotly_white",
    )

    return pn.Column(
        pn.Row(
            pn.indicators.Number(name="Linear R²", value=lr_r2, format="{value:.3f}"),
            pn.indicators.Number(name="Linear RMSE", value=lr_rmse, format="{value:.2f}"),
            pn.indicators.Number(name="Square (Poly) R²", value=poly_r2, format="{value:.3f}"),
            pn.indicators.Number(name="Square (Poly) RMSE", value=poly_rmse, format="{value:.2f}"),
        ),
        pn.pane.Plotly(fig, sizing_mode="stretch_width", height=450),
    )


# ---------------------------------------------------------
# 5. CROSS VALIDATION & K-FOLD MODULE
# ---------------------------------------------------------
@pn.depends(target_var_select.param.value, k_folds_slider.param.value)
def run_kfold_cv(target_col, k_folds):
    ml_df = df[numeric_cols].dropna()
    X = ml_df.drop(columns=[target_col])
    y = ml_df[target_col]

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    model = DecisionTreeRegressor(max_depth=5, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring="r2")

    cv_df = pd.DataFrame({"Fold": [f"Fold {i+1}" for i in range(k_folds)], "R² Score": scores})
    table = pn.widgets.Tabulator(cv_df, disabled=True, show_index=False, height=200)

    fig = px.bar(
        cv_df, x="Fold", y="R² Score", color="R² Score",
        title=f"K-Fold Cross Validation Performance (K={k_folds})",
        color_continuous_scale="Blues", template="plotly_white"
    )
    fig.add_hline(y=scores.mean(), line_dash="dash", annotation_text=f"Mean R²: {scores.mean():.3f}")

    return pn.Column(
        pn.pane.Markdown(f"### K-Fold Cross-Validation Metrics\n- **Mean Score:** `{scores.mean():.4f}` | **Std Dev:** `{scores.std():.4f}`"),
        pn.Row(table, pn.pane.Plotly(fig, sizing_mode="stretch_width", height=350))
    )


# ---------------------------------------------------------
# 6. MODEL COMPLEXITY, LEARNING CURVES & OVERFITTING DIAGNOSTICS
# ---------------------------------------------------------
@pn.depends(target_var_select.param.value, max_depth_slider.param.value)
def get_model_complexity_and_fit_diagnostics(target_col, max_depth_limit):
    ml_df = df[numeric_cols].dropna()
    X = ml_df.drop(columns=[target_col])
    y = ml_df[target_col]

    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    depth_range = np.arange(1, max_depth_limit + 1)

    train_scores, test_scores = validation_curve(
        DecisionTreeRegressor(random_state=42), X, y,
        param_name="max_depth", param_range=depth_range, cv=cv, scoring="r2"
    )

    train_mean, test_mean = train_scores.mean(axis=1), test_scores.mean(axis=1)

    gap = train_mean[-1] - test_mean[-1]
    if train_mean[-1] < 0.5 and test_mean[-1] < 0.5:
        fit_status = "⚠️ **High Bias (Underfitting)**: Model complexity is too low. Increase tree depth or add polynomial features."
        alert_type = "warning"
    elif gap > 0.15:
        fit_status = f"⚠️ **High Variance (Overfitting)**: Large training/validation gap ({gap:.2f}). Regularize parameters or collect more data."
        alert_type = "danger"
    else:
        fit_status = f"✅ **Optimal Fit**: Well balanced model with low variance gap ({gap:.2f})."
        alert_type = "success"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=depth_range, y=train_mean, mode="lines+markers", name="Training R² Score", line=dict(color="#E74C3C")))
    fig.add_trace(go.Scatter(x=depth_range, y=test_mean, mode="lines+markers", name="Validation R² Score", line=dict(color="#2ECC71")))

    fig.update_layout(
        title=f"Model Complexity Graph (Target: {target_col})",
        xaxis_title="Maximum Depth (Complexity)", yaxis_title="R² Score",
        template="plotly_white", yaxis_range=[-0.05, 1.05]
    )

    return pn.Column(
        pn.pane.Alert(fit_status, alert_type=alert_type),
        pn.pane.Plotly(fig, sizing_mode="stretch_width", height=450)
    )


# ---------------------------------------------------------
# 7. SKLEARN GRID SEARCH LAB & SOLUTION
# ---------------------------------------------------------
@pn.depends(target_var_select.param.value, gs_max_depth_range.param.value, gs_min_samples_split.param.value)
def run_grid_search_lab(target_col, depth_range, min_splits):
    if not min_splits:
        min_splits = [2]

    ml_df = df[numeric_cols].dropna()
    X = ml_df.drop(columns=[target_col])
    y = ml_df[target_col]

    param_grid = {
        "max_depth": list(range(depth_range[0], depth_range[1] + 1)),
        "min_samples_split": sorted(int(x) for x in min_splits)
    }

    grid_search = GridSearchCV(
        DecisionTreeRegressor(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        return_train_score=True
    )
    grid_search.fit(X, y)

    results_df = pd.DataFrame(grid_search.cv_results_)
    pivot_df = results_df.pivot(index="param_max_depth", columns="param_min_samples_split", values="mean_test_score")

    fig = px.imshow(
        pivot_df,
        labels=dict(x="Min Samples Split", y="Max Depth", color="CV R² Score"),
        title="Grid Search Parameter Heatmap",
        color_continuous_scale="Viridis",
        text_auto=".3f"
    )

    solution_md = f"""
### 🧪 Grid Search Solution & Optimal Model
- **Best Parameters Selected:** `{grid_search.best_params_}`
- **Best Cross-Validation R² Score:** **{grid_search.best_score_:.4f}**
- **Total Combinations Evaluated:** **{len(results_df)}**
"""

    return pn.Column(
        pn.pane.Markdown(solution_md),
        pn.pane.Plotly(fig, sizing_mode="stretch_width", height=450)
    )


# ---------------------------------------------------------
# 8. PANEL DASHBOARD LAYOUT
# ---------------------------------------------------------
sidebar = pn.Column(
    "## ⚙️ Controls",
    price_tier_select,
    rm_slider,
    pn.layout.Divider(),
    "### 🤖 ML & Grid Search Setup",
    target_var_select,
    k_folds_slider,
    max_depth_slider,
    gs_max_depth_range,
    gs_min_samples_split,
    width=320
)

tabs = pn.Tabs(
    ("📊 Statistical Analytics", pn.Column(
        pn.Row(
            pn.pane.Plotly(pn.bind(get_bar_chart, price_tier_select, rm_slider, target_var_select), height=380),
            pn.pane.Plotly(pn.bind(get_pie_chart, price_tier_select, rm_slider), height=380)
        ),
        pn.Row(
            pn.pane.Plotly(pn.bind(get_scatter_plot, price_tier_select, rm_slider, target_var_select), height=380),
            pn.pane.Plotly(pn.bind(get_heatmap, price_tier_select, rm_slider), height=380)
        ),
        pn.bind(get_table, price_tier_select, rm_slider)
    )),
    ("📈 Linear & Square Regression", run_linear_vs_polynomial_regression),
    ("🔄 Cross-Validation (K-Fold)", run_kfold_cv),
    ("📉 Complexity & Overfitting Diagnostics", get_model_complexity_and_fit_diagnostics),
    ("🔍 Grid Search Lab & Solutions", run_grid_search_lab)
)

template = pn.template.FastListTemplate(
    title="Boston Housing - Advanced ML, CV & Hyperparameter Optimization",
    sidebar=[sidebar],
    main=[tabs],
    accent_base_color="#1f77b4",
    header_background="#1f77b4"
)

template.servable()
