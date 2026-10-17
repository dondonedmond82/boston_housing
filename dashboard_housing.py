"""
Decision Intelligence & Forecasting Dashboard
Compatible with Python 3.10+

Install:
    python -m pip install pandas numpy panel plotly scikit-learn

Run from the folder containing this file:
    panel serve housing_dashboard.py --show

Optional input file:
    housing.csv

Expected examples:
    Target: MEDV, Price, SalePrice, or another numeric column
    Predictors: numeric columns such as RM, LSTAT, PTRATIO
    Time-series: a parseable date column such as Date, Year, Month, or Timestamp

If housing.csv is absent, clearly labeled synthetic demo data is used.
Do not use synthetic demo results for real-world decisions.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import panel as pn
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings("default")
pn.extension("plotly", "tabulator", sizing_mode="stretch_width")

RANDOM_STATE = 42
DATA_PATH = Path("housing.csv")
TARGET_CANDIDATES = ["MEDV", "Price", "price", "SalePrice", "target", "value"]
DATE_NAMES = {"date", "datetime", "timestamp", "year", "month", "period"}


# ---------------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------------
def make_demo_data(n=600):
    """Generate synthetic data for testing the dashboard interface only."""
    rng = np.random.default_rng(RANDOM_STATE)

    rooms = rng.normal(6.2, 0.7, n).clip(3.5, 9.0)
    lstat = rng.uniform(1.5, 38.0, n)
    ptratio = rng.uniform(12.0, 22.0, n)
    noise = rng.normal(0, 35000, n)

    price = (
        470000 + rooms * 72000 - lstat * 10500 - ptratio * 4200 + noise
    ).clip(50000, 1500000)

    dates = pd.date_range("2010-01-01", periods=n, freq="MS")
    trend = np.linspace(0, 180000, n)
    seasonal = 18000 * np.sin(np.arange(n) * 2 * np.pi / 12)
    time_price = (
        320000 + trend + seasonal + rng.normal(0, 22000, n)
    ).clip(100000, None)

    return pd.DataFrame(
        {
            "Date": dates,
            "RM": rooms,
            "LSTAT": lstat,
            "PTRATIO": ptratio,
            "MEDV": price,
            "Demo_Time_Series_Price": time_price,
        }
    )


@pn.cache
def load_data():
    """
    Always return exactly one object: a DataFrame.
    The source description is stored in df.attrs to avoid unpacking errors.
    """
    if DATA_PATH.exists():
        data = pd.read_csv(DATA_PATH)
        data.attrs["source_note"] = (
            f"**Input source:** `{DATA_PATH}`. Check column meanings, units, "
            "data provenance, and quality before operational use."
        )
    else:
        data = make_demo_data()
        data.attrs["source_note"] = (
            f"**DEMO MODE:** `{DATA_PATH}` was not found. Synthetic data is "
            "being used. Forecasts and metrics are demonstrations only."
        )
    return data


# IMPORTANT: Do not unpack load_data() into multiple variables.
raw_df = load_data()
source_note = raw_df.attrs.get("source_note", "Data loaded.")
df = raw_df.copy()

if df.empty:
    raise ValueError("The selected data file contains no rows.")

numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
if not numeric_columns:
    raise ValueError(
        "The data must contain at least one numeric column for predictive modeling."
    )

target_default = next(
    (column for column in TARGET_CANDIDATES if column in numeric_columns),
    numeric_columns[0],
)


def detect_date_column(frame):
    """Find a likely date/time column without assuming a particular column name."""
    candidates = []
    for column in frame.columns:
        name = str(column).lower()
        if name in DATE_NAMES or "date" in name or "time" in name:
            candidates.append(column)

    for column in candidates:
        parsed = pd.to_datetime(frame[column], errors="coerce")
        if parsed.notna().mean() >= 0.70:
            return column

    # Try other object columns as a fallback.
    for column in frame.select_dtypes(include=["object"]).columns:
        parsed = pd.to_datetime(frame[column], errors="coerce")
        if parsed.notna().mean() >= 0.85:
            return column
    return None


detected_date_column = detect_date_column(df)


# ---------------------------------------------------------------------------
# 2. CONTROLS
# ---------------------------------------------------------------------------
target_select = pn.widgets.Select(
    name="Target to predict",
    options=numeric_columns,
    value=target_default,
)

initial_features = [c for c in numeric_columns if c != target_default]
feature_select = pn.widgets.MultiChoice(
    name="Predictor variables",
    options=initial_features,
    value=initial_features,
)

model_select = pn.widgets.Select(
    name="Prediction model",
    options=[
        "Random Forest",
        "Extra Trees",
        "Gradient Boosting",
        "Linear Regression",
        "Ridge Regression",
    ],
    value="Random Forest",
)

test_size = pn.widgets.FloatSlider(
    name="Test data share", start=0.15, end=0.40, step=0.05, value=0.20
)
cv_folds = pn.widgets.IntSlider(
    name="Cross-validation folds", start=3, end=10, step=1, value=5
)
tree_count = pn.widgets.IntSlider(
    name="Number of trees", start=100, end=600, step=50, value=250
)

date_options = ["(No date column)"] + list(df.columns)
date_default = detected_date_column if detected_date_column is not None else "(No date column)"
date_select = pn.widgets.Select(
    name="Date/time column (for forecasting)",
    options=date_options,
    value=date_default,
)
forecast_periods = pn.widgets.IntSlider(
    name="Forecast periods", start=3, end=36, step=1, value=12
)
forecast_freq = pn.widgets.Select(
    name="Forecast frequency",
    options={"Monthly": "MS", "Quarterly": "QS", "Yearly": "YS"},
    value="MS",
)
scenario_pct = pn.widgets.IntSlider(
    name="Change all selected inputs (%)",
    start=-30,
    end=30,
    step=5,
    value=0,
)
show_data = pn.widgets.Checkbox(name="Show first 100 source rows", value=False)


# ---------------------------------------------------------------------------
# 3. SHARED HELPERS
# ---------------------------------------------------------------------------
def empty_figure(title, message):
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template="plotly_white",
        annotations=[
            dict(
                text=message,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=14),
            )
        ],
    )
    return fig


def clean_model_frame(target, features):
    data = df[features + [target]].copy()
    data = data.replace([np.inf, -np.inf], np.nan)
    data[target] = pd.to_numeric(data[target], errors="coerce")
    data = data.dropna(subset=[target])
    for column in features:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def get_features(target, selected_features=None):
    if selected_features is None:
        selected_features = feature_select.value
    features = [
        c for c in selected_features
        if c in numeric_columns and c != target
    ]
    if not features:
        features = [c for c in numeric_columns if c != target]
    return features


def make_model(model_name, n_trees):
    if model_name == "Extra Trees":
        estimator = ExtraTreesRegressor(
            n_estimators=n_trees,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    elif model_name == "Gradient Boosting":
        estimator = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.06,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        )
    elif model_name == "Linear Regression":
        estimator = LinearRegression()
    elif model_name == "Ridge Regression":
        estimator = Ridge(alpha=1.0)
    else:
        estimator = RandomForestRegressor(
            n_estimators=n_trees,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ]
    )


def fit_and_evaluate(target, features, model_name, test_fraction, folds, n_trees):
    """Return final fitted model, split data, metrics, or a readable error."""
    if not features:
        return None, None, None, "Select at least one predictor."

    data = clean_model_frame(target, features)
    if len(data) < 12:
        return (
            None,
            None,
            None,
            "At least 12 usable rows are recommended for model evaluation.",
        )

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_fraction,
        random_state=RANDOM_STATE,
    )

    model = make_model(model_name, n_trees)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, predictions))
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = float(r2_score(y_test, predictions)) if len(y_test) > 1 else np.nan

    # Keep folds valid even for small datasets.
    folds_used = min(folds, len(X_train))
    if folds_used >= 2:
        cv_scores = cross_val_score(
            make_model(model_name, n_trees),
            X_train,
            y_train,
            cv=KFold(
                n_splits=folds_used,
                shuffle=True,
                random_state=RANDOM_STATE,
            ),
            scoring="neg_mean_absolute_error",
        )
        cv_mae = float(-cv_scores.mean())
        cv_sd = float(cv_scores.std())
    else:
        cv_mae, cv_sd = np.nan, np.nan

    # Refit on all available labeled rows for scenarios and future predictions.
    final_model = make_model(model_name, n_trees)
    final_model.fit(X, y)

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "CV_MAE": cv_mae,
        "CV_SD": cv_sd,
        "n": len(data),
    }
    split = (X_train, X_test, y_train, y_test, predictions)
    return final_model, split, metrics, None


# ---------------------------------------------------------------------------
# 4. DECISION SUPPORT & PREDICTIVE MODELING
# ---------------------------------------------------------------------------
@pn.depends(
    target_select.param.value,
    feature_select.param.value,
    model_select.param.value,
    test_size.param.value,
    cv_folds.param.value,
    tree_count.param.value,
    scenario_pct.param.value,
)
def decision_dashboard(
    target,
    selected_features,
    selected_model,
    test_fraction,
    folds,
    n_trees,
    scenario_change,
):
    features = get_features(target, selected_features)
    model, split, metrics, error = fit_and_evaluate(
        target, features, selected_model, test_fraction, folds, n_trees
    )
    if error:
        return pn.pane.Alert(error, alert_type="warning")

    X_train, X_test, y_train, y_test, predictions = split

    metrics_row = pn.Row(
        pn.indicators.Number(
            name="Test R² (higher is generally better)",
            value=metrics["R2"],
            format="{value:.3f}",
        ),
        pn.indicators.Number(
            name="Test MAE (lower is better)",
            value=metrics["MAE"],
            format="{value:,.2f}",
        ),
        pn.indicators.Number(
            name="Test RMSE (lower is better)",
            value=metrics["RMSE"],
            format="{value:,.2f}",
        ),
        pn.indicators.Number(
            name="Cross-validation MAE",
            value=metrics["CV_MAE"],
            format="{value:,.2f}",
        ),
        sizing_mode="stretch_width",
    )

    actual_pred = pd.DataFrame(
        {"Actual": np.asarray(y_test), "Predicted": predictions}
    )
    fig_pred = px.scatter(
        actual_pred,
        x="Actual",
        y="Predicted",
        title="Held-out test set: actual versus predicted",
        template="plotly_white",
    )
    low = float(min(actual_pred["Actual"].min(), actual_pred["Predicted"].min()))
    high = float(max(actual_pred["Actual"].max(), actual_pred["Predicted"].max()))
    fig_pred.add_trace(
        go.Scatter(
            x=[low, high],
            y=[low, high],
            mode="lines",
            name="Ideal prediction",
            line=dict(dash="dash"),
        )
    )
    fig_pred.update_layout(
        xaxis_title=f"Actual {target}",
        yaxis_title=f"Predicted {target}",
    )

    # Permutation importance is measured on held-out data, not training data.
    try:
        importance = permutation_importance(
            make_model(selected_model, n_trees).fit(X_train, y_train),
            X_test,
            y_test,
            n_repeats=5,
            random_state=RANDOM_STATE,
            scoring="r2",
        )
        importance_df = pd.DataFrame(
            {
                "Feature": features,
                "Importance": importance.importances_mean,
            }
        ).sort_values("Importance")
        fig_importance = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Permutation feature importance (held-out data)",
            template="plotly_white",
        )
    except Exception as exc:
        fig_importance = empty_figure(
            "Feature importance",
            f"Importance could not be calculated: {exc}",
        )

    # Scenario uses median observed predictor values as the reference case.
    training_frame = clean_model_frame(target, features)
    baseline = training_frame[features].median(numeric_only=True).to_frame().T
    changed = baseline.copy()
    changed.loc[:, features] = changed[features] * (
        1 + float(scenario_change) / 100.0
    )
    baseline_prediction = float(model.predict(baseline[features])[0])
    scenario_prediction = float(model.predict(changed[features])[0])
    difference = scenario_prediction - baseline_prediction
    difference_pct = (
        difference / abs(baseline_prediction) * 100
        if baseline_prediction != 0
        else np.nan
    )

    scenario_md = f"""
### Decision scenario

The reference case uses the median of each selected predictor. The scenario changes
**all selected predictors together** by **{scenario_change:+.0f}%**.

- **Reference predicted {target}:** {baseline_prediction:,.2f}
- **Scenario predicted {target}:** {scenario_prediction:,.2f}
- **Difference:** {difference:+,.2f} ({difference_pct:+.2f}% where defined)

This is a model-based sensitivity scenario, not a causal estimate. The selected
variables may not be independently changeable in real life.
"""
    model_md = (
        f"**Rows used:** {metrics['n']:,} · "
        f"**Predictors:** {', '.join(features)} · "
        f"**Model:** {selected_model} · "
        f"**CV MAE standard deviation:** {metrics['CV_SD']:,.2f}"
    )

    return pn.Column(
        metrics_row,
        pn.pane.Markdown(model_md),
        pn.Row(
            pn.pane.Plotly(
                fig_pred, height=390, sizing_mode="stretch_width"
            ),
            pn.pane.Plotly(
                fig_importance, height=390, sizing_mode="stretch_width"
            ),
        ),
        pn.pane.Markdown(scenario_md),
        sizing_mode="stretch_width",
    )


# ---------------------------------------------------------------------------
# 5. TIME-SERIES FORECASTING
# ---------------------------------------------------------------------------
@pn.depends(
    target_select.param.value,
    date_select.param.value,
    forecast_periods.param.value,
    forecast_freq.param.value,
)
def time_series_forecast(target, date_column, periods, frequency):
    if date_column == "(No date column)" or date_column not in df.columns:
        return pn.Column(
            pn.pane.Alert(
                "Choose a valid date/time column to enable time-based forecasting.",
                alert_type="info",
            ),
            pn.pane.Markdown(
                "Without dated observations, the dashboard can predict a target "
                "from features but cannot infer a future time sequence."
            ),
        )

    series_df = df[[date_column, target]].copy()
    series_df[date_column] = pd.to_datetime(
        series_df[date_column], errors="coerce"
    )
    series_df[target] = pd.to_numeric(series_df[target], errors="coerce")
    series_df = series_df.dropna().sort_values(date_column)

    if len(series_df) < 12:
        return pn.pane.Alert(
            "At least 12 dated observations are recommended for this basic forecast.",
            alert_type="warning",
        )

    series_df = (
        series_df.groupby(date_column, as_index=False)[target]
        .mean()
        .sort_values(date_column)
    )
    series = series_df.set_index(date_column)[target].sort_index()

    try:
        series = series.resample(frequency).mean().interpolate(
            limit_direction="both"
        )
    except Exception as exc:
        return pn.pane.Alert(
            f"Could not resample the selected dates: {exc}",
            alert_type="warning",
        )

    series = series.dropna()
    if len(series) < 8:
        return pn.pane.Alert(
            "Too few observations at this frequency. Try a finer frequency.",
            alert_type="warning",
        )

    # Chronological backtest: later observations are held out.
    split_index = max(5, int(len(series) * 0.80))
    split_index = min(split_index, len(series) - 1)
    train = series.iloc[:split_index]
    test = series.iloc[split_index:]

    train_x = np.arange(len(train)).reshape(-1, 1)
    test_x = np.arange(len(train), len(series)).reshape(-1, 1)

    # Quadratic trend model. It is intentionally simple and is not seasonal ARIMA.
    trend_model = Pipeline(
        [
            ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
            ("ridge", Ridge(alpha=10.0)),
        ]
    )
    trend_model.fit(train_x, train.values)
    test_predictions = trend_model.predict(test_x)
    backtest_mae = float(mean_absolute_error(test.values, test_predictions))
    backtest_rmse = float(
        np.sqrt(mean_squared_error(test.values, test_predictions))
    )

    # Refit the trend model on all available observations.
    full_x = np.arange(len(series)).reshape(-1, 1)
    trend_model.fit(full_x, series.values)
    future_x = np.arange(
        len(series), len(series) + periods
    ).reshape(-1, 1)
    future_predictions = trend_model.predict(future_x)

    residuals = series.values - trend_model.predict(full_x)
    residual_sd = (
        float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
    )
    horizon = np.arange(1, periods + 1)
    uncertainty = 1.96 * residual_sd * np.sqrt(
        1 + horizon / max(len(series), 1)
    )

    future_dates = pd.date_range(
        start=series.index[-1],
        periods=periods + 1,
        freq=frequency,
    )[1:]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            name="Observed",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=future_predictions,
            mode="lines+markers",
            name="Forecast",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(future_dates) + list(future_dates[::-1]),
            y=list(future_predictions + uncertainty)
            + list((future_predictions - uncertainty)[::-1]),
            fill="toself",
            mode="lines",
            line=dict(width=0),
            name="Approximate uncertainty band",
            opacity=0.2,
        )
    )
    fig.update_layout(
        title=f"{target}: trend forecast",
        xaxis_title="Date",
        yaxis_title=target,
        template="plotly_white",
        legend=dict(orientation="h"),
    )

    forecast_table = pd.DataFrame(
        {
            "Forecast date": future_dates,
            "Forecast": future_predictions,
            "Lower estimate": future_predictions - uncertainty,
            "Upper estimate": future_predictions + uncertainty,
        }
    )
    for column in ["Forecast", "Lower estimate", "Upper estimate"]:
        forecast_table[column] = forecast_table[column].round(2)

    interpretation = f"""
### Forecast quality and interpretation

- **Chronological backtest MAE:** {backtest_mae:,.2f}
- **Chronological backtest RMSE:** {backtest_rmse:,.2f}
- **Historical observations after resampling:** {len(series):,}

The shaded range is an approximate uncertainty band, not a guaranteed confidence
interval. This quadratic trend model does not explicitly model seasonality,
structural breaks, policy changes, or external drivers. Compare it with a
seasonal-naive baseline and domain knowledge before operational use.
"""
    return pn.Column(
        pn.Row(
            pn.indicators.Number(
                name="Backtest MAE",
                value=backtest_mae,
                format="{value:,.2f}",
            ),
            pn.indicators.Number(
                name="Backtest RMSE",
                value=backtest_rmse,
                format="{value:,.2f}",
            ),
            pn.indicators.Number(
                name="Forecast horizon",
                value=periods,
                format="{value:.0f} periods",
            ),
        ),
        pn.pane.Plotly(fig, height=460, sizing_mode="stretch_width"),
        pn.pane.Markdown(interpretation),
        pn.widgets.Tabulator(
            forecast_table,
            show_index=False,
            pagination="local",
            page_size=12,
            height=280,
        ),
    )


# ---------------------------------------------------------------------------
# 6. DATA QUALITY & DESCRIPTIVE ANALYTICS
# ---------------------------------------------------------------------------
@pn.depends(target_select.param.value, show_data.param.value)
def data_quality(target, display_source_data):
    missing = df.isna().sum().rename("Missing values").to_frame()
    missing["Missing %"] = (
        missing["Missing values"] / max(len(df), 1) * 100
    ).round(2)
    missing.index.name = "Column"
    missing_table = missing.reset_index()

    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] >= 2:
        correlation_fig = px.imshow(
            numeric_df.corr(),
            text_auto=".2f",
            aspect="auto",
            title="Numeric feature correlation",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            template="plotly_white",
        )
    else:
        correlation_fig = empty_figure(
            "Numeric feature correlation",
            "At least two numeric columns are required.",
        )

    target_stats = (
        pd.to_numeric(df[target], errors="coerce")
        .describe()
        .rename_axis("Statistic")
        .reset_index(name="Value")
    )

    parts = [
        pn.pane.Markdown(
            f"""### Data quality overview

- **Rows:** {len(df):,}
- **Columns:** {len(df.columns):,}
- **Duplicate rows:** {int(df.duplicated().sum()):,}
- **Selected target:** `{target}`
"""
        ),
        pn.Row(
            pn.Column(
                "#### Missing values",
                pn.widgets.Tabulator(
                    missing_table,
                    show_index=False,
                    height=280,
                    pagination="local",
                    page_size=10,
                ),
            ),
            pn.Column(
                "#### Target summary",
                pn.widgets.Tabulator(
                    target_stats,
                    show_index=False,
                    height=280,
                ),
            ),
        ),
        pn.pane.Plotly(
            correlation_fig, height=450, sizing_mode="stretch_width"
        ),
    ]

    if display_source_data:
        parts.append(
            pn.Column(
                "#### First 100 source rows",
                pn.widgets.Tabulator(
                    df.head(100),
                    show_index=False,
                    height=320,
                    pagination="local",
                    page_size=10,
                ),
            )
        )
    else:
        parts.append(
            pn.pane.Markdown(
                "Enable **Show first 100 source rows** in the sidebar to inspect the data."
            )
        )

    return pn.Column(*parts, sizing_mode="stretch_width")


# ---------------------------------------------------------------------------
# 7. DASHBOARD LAYOUT
# ---------------------------------------------------------------------------
sidebar = pn.Column(
    pn.pane.Markdown("## Decision controls"),
    pn.pane.Markdown(source_note),
    target_select,
    feature_select,
    model_select,
    test_size,
    cv_folds,
    tree_count,
    pn.layout.Divider(),
    pn.pane.Markdown("### Forecast settings"),
    date_select,
    forecast_periods,
    forecast_freq,
    pn.layout.Divider(),
    pn.pane.Markdown("### Scenario analysis"),
    scenario_pct,
    show_data,
    width=350,
)

tabs = pn.Tabs(
    ("Decision Support & Model", decision_dashboard),
    ("Time-Series Forecast", time_series_forecast),
    ("Data Quality & Drivers", data_quality),
    dynamic=True,
)

template = pn.template.FastListTemplate(
    title="Decision Intelligence | Predictive Analytics & Forecasting",
    sidebar=[sidebar],
    main=[
        pn.pane.Markdown(
            "Evaluate model performance, explore influential predictors, test "
            "scenarios, inspect data quality, and forecast dated series. "
            "Predictions support—but do not replace—professional judgment."
        ),
        tabs,
    ],
    accent_base_color="#1f77b4",
    header_background="#1f4e79",
)

template.servable()
