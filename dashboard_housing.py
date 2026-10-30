import os
import warnings
import numpy as np
import pandas as pd
import panel as pn
import plotly.express as px
import plotly.graph_objects as go

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (
    ShuffleSplit,
    KFold,
    train_test_split,
    validation_curve,
    cross_val_score,
    GridSearchCV,
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

pn.extension(
    "plotly",
    "tabulator",
    sizing_mode="stretch_width",
    raw_css=[
        """
        /* Force the Bokeh/Panel tab headers to wrap into two visible rows. */
        .bk-tabs-header,
        .bk-headers {
            height: auto !important;
            min-height: 0 !important;
            max-height: none !important;
            overflow: visible !important;
        }

        .bk-headers {
            display: flex !important;
            flex-flow: row wrap !important;
            align-content: flex-start !important;
            align-items: stretch !important;
            width: 100% !important;
            gap: 4px !important;
        }

        .bk-headers > .bk-header {
            box-sizing: border-box !important;
            flex: 0 0 calc(25% - 4px) !important;
            width: calc(25% - 4px) !important;
            min-width: 0 !important;
            max-width: calc(25% - 4px) !important;
            margin: 0 !important;
        }

        .bk-header .bk-tab {
            box-sizing: border-box !important;
            display: block !important;
            width: 100% !important;
            height: auto !important;
            min-height: 38px !important;
            padding: 8px 5px !important;
            white-space: normal !important;
            overflow-wrap: anywhere !important;
            text-align: center !important;
            line-height: 1.25 !important;
        }

        /* Keep the tab content below both header rows, not over them. */
        .bk-tabs-header + .bk-tabs-content,
        .bk-tabs-content {
            min-height: 0 !important;
        }
        """
    ],
)


# ---------------------------------------------------------
# 1. DATA LOADING & PREPROCESSING
# ---------------------------------------------------------
def load_data():
    """Load the housing data and always build the dashboard's derived columns."""
    if os.path.exists("housing.csv"):
        df = pd.read_csv("housing.csv")
        # Clean CSV headers (including whitespace/BOM) and standardize expected names.
        df.columns = (
            df.columns.astype(str)
            .str.replace("\ufeff", "", regex=False)
            .str.strip()
            .str.upper()
        )
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

    required_cols = ["RM", "LSTAT", "PTRATIO", "MEDV"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            "housing.csv is missing required columns: " + ", ".join(missing_cols)
            + ". Expected columns: RM, LSTAT, PTRATIO, MEDV."
        )

    # Ensure the fields used by charts and machine-learning models are numeric.
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required_cols).copy()
    if df.empty:
        raise ValueError("housing.csv contains no valid rows for RM, LSTAT, PTRATIO and MEDV.")

    # Recreate derived columns on every load; never depend on them being in the CSV.
    df["Price_Tier"] = pd.qcut(
        df["MEDV"].rank(method="first"),
        q=3,
        labels=["Affordable", "Mid-Range", "Luxury"],
    )

    df["Housing_Summary_Text"] = (
        "Property record with " + df["RM"].round(1).astype(str) + " rooms, "
        "LSTAT value of " + df["LSTAT"].round(1).astype(str) + "% , and a pupil-teacher ratio of "
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
    if "Price_Tier" not in df.columns:
        raise ValueError(
            "Price_Tier is missing. Reload the data using load_data() so the derived column is created."
        )
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
# 8. EXPLORATORY DATA ANALYSIS (EDA) & DATA-DRIVEN RULES
# ---------------------------------------------------------
# Dataset-specific guide:
# - RM: average number of rooms per dwelling.
# - LSTAT: percentage indicator supplied in the dataset.
# - PTRATIO: pupil-teacher ratio.
# - MEDV: target variable representing the recorded median property value.
#
# The rules below are calculated from the uploaded housing.csv file at runtime.
# Correlation describes association, not causation. IQR outliers are flags for
# investigation; they are not automatically deleted or treated as errors.

EDA_FEATURES = ["RM", "LSTAT", "PTRATIO"]
EDA_TARGET = "MEDV"


def _correlation_strength(value):
    """Describe the magnitude of a Pearson correlation."""
    magnitude = abs(float(value))
    if magnitude >= 0.70:
        return "strong"
    if magnitude >= 0.50:
        return "moderate"
    if magnitude >= 0.30:
        return "weak"
    return "very weak"


def build_exploratory_report():
    """Create a plain-language report from the current dataset's statistics."""
    data = df[numeric_cols].copy()
    corr = data.corr(numeric_only=True)
    lines = [
        "## Dataset overview",
        f"- **Observations:** {len(data):,}",
        f"- **Numerical variables:** {', '.join(numeric_cols)}",
        f"- **Missing values in analysis columns:** {int(data.isna().sum().sum())}",
        f"- **Duplicate rows:** {int(data.duplicated().sum())}",
        "",
        "## Descriptive interpretation",
        f"- **RM:** average {data['RM'].mean():.2f} rooms; range "
        f"{data['RM'].min():.2f}–{data['RM'].max():.2f}.",
        f"- **LSTAT:** average {data['LSTAT'].mean():.2f}; range "
        f"{data['LSTAT'].min():.2f}–{data['LSTAT'].max():.2f}.",
        f"- **PTRATIO:** average {data['PTRATIO'].mean():.2f}; range "
        f"{data['PTRATIO'].min():.2f}–{data['PTRATIO'].max():.2f}.",
        f"- **MEDV:** average {data['MEDV'].mean():,.2f}; median "
        f"{data['MEDV'].median():,.2f}; range "
        f"{data['MEDV'].min():,.2f}–{data['MEDV'].max():,.2f}.",
        "",
        "## Exploratory rules and findings",
    ]

    # Interpret relationships with MEDV based on the actual uploaded data.
    for feature in EDA_FEATURES:
        value = float(corr.loc[feature, EDA_TARGET])
        strength = _correlation_strength(value)
        direction = "positive" if value > 0 else "negative" if value < 0 else "approximately zero"
        if value > 0:
            interpretation = "higher values tend to occur with higher MEDV values"
        elif value < 0:
            interpretation = "higher values tend to occur with lower MEDV values"
        else:
            interpretation = "there is no linear direction in this sample"
        lines.append(
            f"- **{feature} vs MEDV:** {direction} {strength} association "
            f"(Pearson r = {value:.3f}); {interpretation}."
        )

    # Explain outlier flags using the standard 1.5 x IQR screening rule.
    lines.extend(["", "### Potential outliers (1.5 × IQR screening)"])
    for column in numeric_cols:
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((data[column] < lower) | (data[column] > upper)).sum())
        lines.append(
            f"- **{column}:** {count} observation(s) outside "
            f"[{lower:,.2f}, {upper:,.2f}]. Review these records; do not remove "
            "them without checking their meaning."
        )

    lines.extend([
        "",
        "### Practical analysis rules",
        "1. Check missing values and data types before fitting models.",
        "2. Inspect histograms and box plots to understand skewness and unusual values.",
        "3. Use the correlation matrix to identify linear associations, not causal effects.",
        "4. Keep `MEDV` as the prediction target when the goal is house-value estimation.",
        "5. Use `RM`, `LSTAT`, and `PTRATIO` as predictors for the house-value task.",
        "6. Split training and test data before fitting preprocessing or prediction models.",
        "7. Compare models with MAE and RMSE (lower is better) and R² (higher is generally better).",
        "8. Check cross-validation and overfitting diagnostics before selecting a final model.",
        "",
        "**Important:** the dataset contains four numerical columns only. It does not provide "
        "location, property age, property type, or a documented currency/unit field. "
        "Do not infer those attributes from the numeric values.",
        "",
        "**Interpretation caution:** correlation and model feature importance indicate "
        "patterns in this dataset; they do not establish that one variable causes changes in another."
    ])
    return "\n".join(lines)


def build_eda_distribution_plots():
    """Show one histogram and box-plot marginal for each numeric variable."""
    plots = []
    for column in numeric_cols:
        fig = px.histogram(
            df, x=column, marginal="box", nbins=30,
            title=f"Distribution of {column}",
            template="plotly_white"
        )
        fig.update_layout(height=330, margin=dict(l=25, r=20, t=55, b=25))
        plots.append(pn.pane.Plotly(fig, sizing_mode="stretch_width", height=330))
    return pn.Column(
        pn.Row(plots[0], plots[1], sizing_mode="stretch_width"),
        pn.Row(plots[2], plots[3], sizing_mode="stretch_width"),
        sizing_mode="stretch_width"
    )


def build_eda_correlation_plot():
    """Display Pearson correlations for the four supplied numeric variables."""
    corr = df[numeric_cols].corr()
    fig = px.imshow(
        corr, text_auto=".2f", zmin=-1, zmax=1,
        color_continuous_scale="RdBu_r",
        title="Pearson Correlation Matrix",
        labels={"color": "Correlation"},
        template="plotly_white"
    )
    fig.update_layout(height=480)
    return pn.pane.Plotly(fig, sizing_mode="stretch_width", height=480)


def build_eda_relationship_plots():
    """Show the target against each supplied explanatory variable."""
    plots = []
    for feature in EDA_FEATURES:
        fig = px.scatter(
            df, x=feature, y=EDA_TARGET,
            title=f"{feature} vs {EDA_TARGET}",
            opacity=0.70, template="plotly_white"
        )
        # Add a simple least-squares trend line without requiring statsmodels.
        slope, intercept = np.polyfit(df[feature].to_numpy(), df[EDA_TARGET].to_numpy(), 1)
        x_line = np.linspace(float(df[feature].min()), float(df[feature].max()), 100)
        fig.add_trace(go.Scatter(
            x=x_line, y=slope * x_line + intercept,
            mode="lines", name="Linear trend"
        ))
        fig.update_layout(height=370)
        plots.append(pn.pane.Plotly(fig, sizing_mode="stretch_width", height=370))
    return pn.Column(
        pn.Row(plots[0], plots[1], sizing_mode="stretch_width"),
        pn.Row(plots[2], sizing_mode="stretch_width"),
        sizing_mode="stretch_width"
    )


def build_eda_summary_table():
    """Return descriptive statistics, including quartiles and skewness."""
    summary = df[numeric_cols].describe().T.reset_index()
    summary = summary.rename(columns={"index": "Variable"})
    summary["skewness"] = df[numeric_cols].skew().values
    return pn.widgets.Tabulator(
        summary.round(3), show_index=False, height=220,
        sizing_mode="stretch_width"
    )


exploratory_analysis_panel = pn.Column(
    pn.pane.Markdown(
        "# Exploratory Data Analysis (EDA)\n"
        "This section examines the uploaded `housing.csv` before modelling. "
        "The written findings and screening rules are recalculated from the data."
    ),
    pn.pane.Markdown(build_exploratory_report()),
    "## Descriptive statistics",
    build_eda_summary_table(),
    "## Variable distributions and potential outliers",
    build_eda_distribution_plots(),
    "## Relationships with the target variable (MEDV)",
    build_eda_relationship_plots(),
    "## Correlation analysis",
    build_eda_correlation_plot(),
    sizing_mode="stretch_width"
)


# ---------------------------------------------------------
# 9. MACHINE-LEARNING MODEL COMPARISON & FEATURE IMPORTANCE
# ---------------------------------------------------------
def _build_ml_models():
    """Return a collection of regression algorithms for benchmark comparison."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=None, min_samples_split=2,
            random_state=42, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42
        ),
        "K-Nearest Neighbors": make_pipeline(
            StandardScaler(), KNeighborsRegressor(n_neighbors=5)
        ),
        "Support Vector Regression": make_pipeline(
            StandardScaler(), SVR(kernel="rbf", C=100, epsilon=0.1)
        ),
    }


@pn.depends(target_var_select.param.value)
def run_ml_model_comparison(target_col):
    """Train and compare multiple ML regressors using one consistent holdout split."""
    ml_df = df[numeric_cols].dropna()
    X = ml_df.drop(columns=[target_col])
    y = ml_df[target_col]

    if len(ml_df) < 10:
        return pn.pane.Alert(
            "At least 10 valid rows are required for model comparison.",
            alert_type="warning"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rows = []
    predictions = {}
    fitted_models = {}

    for model_name, model in _build_ml_models().items():
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            predictions[model_name] = pred
            fitted_models[model_name] = model
            rows.append({
                "Model": model_name,
                "R²": r2_score(y_test, pred),
                "MAE": mean_absolute_error(y_test, pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            })
        except Exception as exc:
            rows.append({
                "Model": model_name,
                "R²": np.nan,
                "MAE": np.nan,
                "RMSE": np.nan,
                "Error": str(exc)[:120],
            })

    results = pd.DataFrame(rows).sort_values("RMSE", na_position="last")
    valid_results = results.dropna(subset=["RMSE"])
    if valid_results.empty:
        return pn.pane.Alert(
            "No model could be trained. Check the input data and installed scikit-learn version.",
            alert_type="danger"
        )

    best_name = valid_results.iloc[0]["Model"]
    best_model = fitted_models[best_name]

    metrics_table = pn.widgets.Tabulator(
        results.round(4), show_index=False, height=300,
        sizing_mode="stretch_width"
    )

    comparison_fig = px.bar(
        valid_results, x="Model", y="RMSE", color="Model",
        title="Model Comparison — Lower RMSE Is Better",
        template="plotly_white"
    )
    comparison_fig.update_layout(
        xaxis_tickangle=-25, showlegend=False, margin=dict(b=80, t=55)
    )

    pred_fig = go.Figure()
    for model_name, pred in predictions.items():
        pred_fig.add_trace(go.Scatter(
            x=y_test, y=pred, mode="markers", name=model_name
        ))
    low, high = float(y_test.min()), float(y_test.max())
    pred_fig.add_trace(go.Scatter(
        x=[low, high], y=[low, high], mode="lines",
        name="Ideal prediction", line=dict(dash="dash", color="black")
    ))
    pred_fig.update_layout(
        title=f"Actual vs Predicted Values — {target_col}",
        xaxis_title=f"Actual {target_col}",
        yaxis_title=f"Predicted {target_col}",
        template="plotly_white"
    )

    # Tree-based models expose feature_importances_; linear models expose coefficients.
    importance = None
    importance_title = None
    if hasattr(best_model, "feature_importances_"):
        importance = best_model.feature_importances_
        importance_title = f"Feature Importance — {best_name}"
    elif hasattr(best_model, "named_steps"):
        final_estimator = list(best_model.named_steps.values())[-1]
        if hasattr(final_estimator, "coef_"):
            importance = np.abs(np.ravel(final_estimator.coef_))
            importance_title = f"Absolute Coefficient Importance — {best_name}"
    elif hasattr(best_model, "coef_"):
        importance = np.abs(np.ravel(best_model.coef_))
        importance_title = f"Absolute Coefficient Importance — {best_name}"

    components = [
        pn.pane.Markdown(
            f"## Best model: **{best_name}**\n"
            f"- **R²:** {valid_results.iloc[0]['R²']:.4f}\n"
            f"- **MAE:** {valid_results.iloc[0]['MAE']:,.4f}\n"
            f"- **RMSE:** {valid_results.iloc[0]['RMSE']:,.4f}\n\n"
            "Models use the same 80/20 train-test split (random_state=42)."
        ),
        metrics_table,
        pn.pane.Plotly(comparison_fig, sizing_mode="stretch_width", height=400),
        pn.pane.Plotly(pred_fig, sizing_mode="stretch_width", height=450),
    ]

    if importance is not None:
        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importance
        }).sort_values("Importance", ascending=True)
        importance_fig = px.bar(
            importance_df, x="Importance", y="Feature", orientation="h",
            title=importance_title, template="plotly_white"
        )
        components.append(
            pn.pane.Plotly(importance_fig, sizing_mode="stretch_width", height=350)
        )
    else:
        components.append(pn.pane.Markdown(
            "Feature importance is not directly available for the selected best model."
        ))

    return pn.Column(*components, sizing_mode="stretch_width")


# ---------------------------------------------------------
# 10. INTERACTIVE HOUSE VALUE PREDICTION
# ---------------------------------------------------------
prediction_model_select = pn.widgets.Select(
    name="Prediction Algorithm",
    options=[
        "Random Forest", "Gradient Boosting", "Extra Trees",
        "Decision Tree", "Linear Regression"
    ],
    value="Random Forest",
    width=220,
)
prediction_rm = pn.widgets.FloatInput(
    name="Rooms (RM)", value=float(df["RM"].median()),
    step=0.1, width=220
)
prediction_lstat = pn.widgets.FloatInput(
    name="Lower Status Population (LSTAT %)",
    value=float(df["LSTAT"].median()), step=0.1, width=220
)
prediction_ptratio = pn.widgets.FloatInput(
    name="Pupil-Teacher Ratio (PTRATIO)",
    value=float(df["PTRATIO"].median()), step=0.1, width=220
)


def predict_house_value(model_name, rm, lstat, ptratio):
    features = df[["RM", "LSTAT", "PTRATIO"]]
    target = df["MEDV"]
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Linear Regression": LinearRegression(),
    }
    model = models[model_name]
    model.fit(features, target)
    input_row = pd.DataFrame(
        [[rm, lstat, ptratio]], columns=["RM", "LSTAT", "PTRATIO"]
    )
    predicted_value = float(model.predict(input_row)[0])

    return pn.pane.Markdown(
        f"## Estimated House Value\n"
        f"### ${predicted_value:,.2f}\n"
        f"**Algorithm:** {model_name}\n\n"
        "This is a model estimate based on the available dataset, not a formal property valuation."
    )


prediction_panel = pn.Column(
    "## 🏠 Predict a House Value",
    prediction_model_select,
    prediction_rm,
    prediction_lstat,
    prediction_ptratio,
    pn.bind(
        predict_house_value,
        prediction_model_select,
        prediction_rm,
        prediction_lstat,
        prediction_ptratio,
    ),
    sizing_mode="stretch_width",
)


# ---------------------------------------------------------
# 11. PANEL DASHBOARD LAYOUT
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
    pn.layout.Divider(),
    "### 🏠 House Value Prediction",
    prediction_model_select,
    prediction_rm,
    prediction_lstat,
    prediction_ptratio,
    width=240,
    min_width=220,
    max_width=240,
    sizing_mode="fixed",
    styles={"padding": "8px"},
    margin=(5, 5, 5, 5)
)

# Keep sidebar widgets within the reduced sidebar width.
for widget in (
    price_tier_select,
    rm_slider,
    target_var_select,
    k_folds_slider,
    max_depth_slider,
    gs_max_depth_range,
    gs_min_samples_split,
    prediction_model_select,
    prediction_rm,
    prediction_lstat,
    prediction_ptratio,
):
    widget.width = 220
    widget.sizing_mode = "fixed"

tabs = pn.Tabs(
    ("🔎 Exploratory Data Analysis", exploratory_analysis_panel),
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
    ("🔍 Grid Search Lab & Solutions", run_grid_search_lab),
    ("🤖 ML Model Comparison", run_ml_model_comparison),
    ("🏠 House Value Prediction", prediction_panel)
)

template = pn.template.FastListTemplate(
    title="Housing Data Analytics - EDA, Statistics & Machine Learning",
    sidebar=[sidebar],
    main=[tabs],
    accent_base_color="#1f77b4",
    header_background="#1f77b4"
)

template.servable()
