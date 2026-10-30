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
    KPli,
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
# 1. CHARGEMENT ET PRÉTRAITEMENT DES DONNÉES
# ---------------------------------------------------------
def load_data():
    """Charge les données immobilières et crée systématiquement les colonnes dérivées du tableau de bord."""
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
        labels=["Abordable", "Gamme intermédiaire", "Luxe"],
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
TIER_ORDER = ["Abordable", "Gamme intermédiaire", "Luxe"]


# ---------------------------------------------------------
# 2. CONTRÔLES ET COMPOSANTS INTERACTIFS
# ---------------------------------------------------------
price_tier_select = pn.widgets.MultiSelect(
    name="Sélectionner les catégories de prix",
    options=TIER_ORDER,
    value=TIER_ORDER,
)

rm_slider = pn.widgets.RangeSlider(
    name="Filtrer par nombre de pièces (RM)",
    start=float(df["RM"].min()),
    end=float(df["RM"].max()),
    value=(float(df["RM"].min()), float(df["RM"].max())),
    step=0.1,
)

target_var_select = pn.widgets.Select(
    name="Variable cible de l’apprentissage automatique", options=numeric_cols, value="MEDV"
)

k_folds_slider = pn.widgets.IntSlider(
    name="Nombre de plis de validation croisée (K)", start=2, end=10, step=1, value=5
)

max_depth_slider = pn.widgets.IntSlider(
    name="Profondeur maximale de l’arbre de décision (complexité)", start=1, end=15, step=1, value=10
)

# Interactive Grid Search Sliders
gs_max_depth_range = pn.widgets.IntRangeSlider(
    name="Plage de profondeur pour la recherche sur grille", start=1, end=15, value=(1, 10), step=1
)

gs_min_samples_split = pn.widgets.MultiChoice(
    name="Nombre minimal d’échantillons pour la séparation",
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
    """Graphique/message standard affiché lorsqu’un filtre ne renvoie aucune ligne."""
    fig = go.Figure()
    fig.update_layout(
        title=f"{title} — aucune donnée pour les filtres actuels",
        annotations=[dict(text="Aucune ligne ne correspond aux filtres sélectionnés",
                           xref="paper", yref="paper", showarrow=False, font=dict(size=14))],
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------
# 3. GÉNÉRATEURS DE COMPOSANTS STATISTIQUES
# ---------------------------------------------------------
def get_table(tiers, rm_range):
    filtered = filter_dataframe(tiers, rm_range)
    if filtered.empty:
        return pn.pane.Markdown("**Aucune donnée pour les filtres actuels.**")
    stats = filtered[numeric_cols].describe().T[["count", "mean", "std", "min", "50%", "max"]]
    stats = stats.reset_index().rename(columns={"index": "Indicator", "50%": "median"})
    return pn.widgets.Tabulator(stats, pagination="remote", page_size=10, height=250)


def get_pie_chart(tiers, rm_range):
    filtered = filter_dataframe(tiers, rm_range)
    if filtered.empty:
        return _empty_notice("Répartition des catégories de prix")
    counts = filtered["Price_Tier"].value_counts().reset_index()
    fig = px.pie(
        counts, values="count", names="Price_Tier",
        title="Répartition des catégories de prix",
        hole=0.4, color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig


def get_bar_chart(tiers, rm_range, target_col="MEDV"):
    filtered = filter_dataframe(tiers, rm_range)
    if filtered.empty:
        return _empty_notice(f"Moyenne {target_col} par catégorie")
    avg_df = filtered.groupby("Price_Tier", observed=True)[target_col].mean().reset_index()
    fig = px.bar(
        avg_df, x="Price_Tier", y=target_col,
        title=f"Moyenne {target_col} par catégorie",
        color="Price_Tier", template="plotly_white"
    )
    return fig


def get_scatter_plot(tiers, rm_range, target_col="MEDV"):
    filtered = filter_dataframe(tiers, rm_range)
    if filtered.empty:
        return _empty_notice(f"LSTAT par rapport à {target_col}")
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
        return _empty_notice("Matrice de corrélation des variables")
    corr = filtered[numeric_cols].corr()
    fig = px.imshow(
        corr, text_auto=".2f",
        title="Matrice de corrélation des variables",
        color_continuous_scale="RdBu_r"
    )
    return fig


# ---------------------------------------------------------
# 4. MODULE DE PRÉDICTION PAR RÉGRESSION LINÉAIRE ET POLYNOMIALE
# ---------------------------------------------------------
@pn.depends(target_var_select.param.value)
def run_linear_vs_polynomial_regression(target_col):
    ml_df = df[numeric_cols].dropna()
    X = ml_df.drop(columns=[target_col])
    y = ml_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. Régression linéaire
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

    # Valeur réelle par rapport à Valeur prédite Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test, y=lr_preds, mode="markers", name="Régression linéaire"))
    fig.add_trace(go.Scatter(x=y_test, y=poly_preds, mode="markers", name="Polynomiale (degré 2)"))
    fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
                             mode="lines", name="Ajustement idéal", line=dict(dash="dash", color="black")))

    fig.update_layout(
        title=f"Prédictions de la régression par rapport aux valeurs réelles ({target_col})",
        xaxis_title=f"Valeur réelle {target_col}",
        yaxis_title=f"Valeur prédite {target_col}",
        template="plotly_white",
    )

    return pn.Column(
        pn.Row(
            pn.indicators.Number(name="R² linéaire", value=lr_r2, format="{value:.3f}"),
            pn.indicators.Number(name="RMSE linéaire", value=lr_rmse, format="{value:.2f}"),
            pn.indicators.Number(name="R² polynomial", value=poly_r2, format="{value:.3f}"),
            pn.indicators.Number(name="RMSE polynomial", value=poly_rmse, format="{value:.2f}"),
        ),
        pn.pane.Plotly(fig, sizing_mode="stretch_width", height=450),
    )


# ---------------------------------------------------------
# 5. MODULE DE VALIDATION CROISÉE ET K-FOLD
# ---------------------------------------------------------
@pn.depends(target_var_select.param.value, k_folds_slider.param.value)
def run_kfold_cv(target_col, k_folds):
    ml_df = df[numeric_cols].dropna()
    X = ml_df.drop(columns=[target_col])
    y = ml_df[target_col]

    kf = KPli(n_splits=k_folds, shuffle=True, random_state=42)
    model = DecisionTreeRegressor(max_depth=5, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring="r2")

    cv_df = pd.DataFrame({"Pli": [f"Pli {i+1}" for i in range(k_folds)], "Score R²": scores})
    table = pn.widgets.Tabulator(cv_df, disabled=True, show_index=False, height=200)

    fig = px.bar(
        cv_df, x="Pli", y="Score R²", color="Score R²",
        title=f"Performance de la validation croisée K-Pli (K={k_folds})",
        color_continuous_scale="Blues", template="plotly_white"
    )
    fig.add_hline(y=scores.mean(), line_dash="dash", annotation_text=f"Moyenne R²: {scores.mean():.3f}")

    return pn.Column(
        pn.pane.Markdown(f"### K-Pli Cross-Validation Metrics\n- **Moyenne Score:** `{scores.mean():.4f}` | **Écart-type :** `{scores.std():.4f}`"),
        pn.Row(table, pn.pane.Plotly(fig, sizing_mode="stretch_width", height=350))
    )


# ---------------------------------------------------------
# 6. COMPLEXITÉ DU MODÈLE, COURBES D’APPRENTISSAGE ET DIAGNOSTIC DU SURAPPRENTISSAGE
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
        fit_status = "⚠️ **Biais élevé (sous-apprentissage)**: La complexité du modèle est trop faible. Augmentez la profondeur de l’arbre ou ajoutez des variables polynomiales."
        alert_type = "warning"
    elif gap > 0.15:
        fit_status = f"⚠️ **Variance élevée (surapprentissage)**: Écart important entre entraînement et validation ({gap:.2f}). Régularisez les paramètres ou collectez davantage de données."
        alert_type = "danger"
    else:
        fit_status = f"✅ **Ajustement optimal**: Modèle bien équilibré avec un faible écart de variance ({gap:.2f})."
        alert_type = "success"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=depth_range, y=train_mean, mode="lines+markers", name="Training Score R²", line=dict(color="#E74C3C")))
    fig.add_trace(go.Scatter(x=depth_range, y=test_mean, mode="lines+markers", name="Validation Score R²", line=dict(color="#2ECC71")))

    fig.update_layout(
        title=f"Graphique de complexité du modèle (cible : {target_col})",
        xaxis_title="Profondeur maximale (complexité)", yaxis_title="Score R²",
        template="plotly_white", yaxis_range=[-0.05, 1.05]
    )

    return pn.Column(
        pn.pane.Alert(fit_status, alert_type=alert_type),
        pn.pane.Plotly(fig, sizing_mode="stretch_width", height=450)
    )


# ---------------------------------------------------------
# 7. LABORATOIRE DE RECHERCHE SUR GRILLE SKLEARN ET RÉSULTATS
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
        labels=dict(x="Nombre minimal d’échantillons pour la séparation", y="Profondeur maximale", color="CV Score R²"),
        title="Carte thermique des paramètres de recherche sur grille",
        color_continuous_scale="Viridis",
        text_auto=".3f"
    )

    solution_md = f"""
### 🧪 Résultat de la recherche sur grille et modèle optimal
- **Meilleurs paramètres sélectionnés :** `{grid_search.best_params_}`
- **Best Cross-Validation Score R²:** **{grid_search.best_score_:.4f}**
- **Nombre total de combinaisons évaluées :** **{len(results_df)}**
"""

    return pn.Column(
        pn.pane.Markdown(solution_md),
        pn.pane.Plotly(fig, sizing_mode="stretch_width", height=450)
    )





# ---------------------------------------------------------
# 8. ANALYSE EXPLORATOIRE DES DONNÉES (AED) ET RÈGLES FONDÉES SUR LES DONNÉES
# ---------------------------------------------------------
# Guide spécifique au jeu de données :
# - RM: moyenne number of rooms per dwelling.
# - LSTAT: indicateur en pourcentage fourni dans le jeu de données.
# - PTRATIO: ratio élèves-enseignant.
# - MEDV: variable cible représentant la valeur médiane enregistrée des biens immobiliers.
#
# Les règles ci-dessous sont calculées à l’exécution à partir du fichier housing.csv fourni.
# Corrélation describes association, not causation. IQR outliers are flags for
# examiner ; elles ne sont pas automatiquement supprimées ni considérées comme des erreurs.

EDA_FEATURES = ["RM", "LSTAT", "PTRATIO"]
EDA_TARGET = "MEDV"


def _correlation_strength(value):
    """Décrit l’intensité d’une corrélation de Pearson."""
    magnitude = abs(float(value))
    if magnitude >= 0.70:
        return "forte"
    if magnitude >= 0.50:
        return "modérée"
    if magnitude >= 0.30:
        return "faible"
    return "very faible"


def build_exploratory_report():
    """Crée un rapport en langage clair à partir des statistiques du jeu de données actuel."""
    data = df[numeric_cols].copy()
    corr = data.corr(numeric_only=True)
    lines = [
        "## Aperçu du jeu de données",
        f"- **Observations :** {len(data):,}",
        f"- **Variables numériques :** {', '.join(numeric_cols)}",
        f"- **Valeurs manquantes dans les colonnes analysées :** {int(data.isna().sum().sum())}",
        f"- **Lignes en double :** {int(data.duplicated().sum())}",
        "",
        "## Interprétation descriptive",
        f"- **RM:** moyenne {data['RM'].mean():.2f} pièces ; intervalle "
        f"{data['RM'].min():.2f}–{data['RM'].max():.2f}.",
        f"- **LSTAT:** moyenne {data['LSTAT'].mean():.2f} ; intervalle "
        f"{data['LSTAT'].min():.2f}–{data['LSTAT'].max():.2f}.",
        f"- **PTRATIO:** moyenne {data['PTRATIO'].mean():.2f} ; intervalle "
        f"{data['PTRATIO'].min():.2f}–{data['PTRATIO'].max():.2f}.",
        f"- **MEDV:** moyenne {data['MEDV'].mean():,.2f}; median "
        f"{data['MEDV'].median():,.2f} ; intervalle "
        f"{data['MEDV'].min():,.2f}–{data['MEDV'].max():,.2f}.",
        "",
        "## Règles et résultats de l’analyse exploratoire",
    ]

    # Interpret relationships with MEDV based on the actual uploaded data.
    for feature in EDA_FEATURES:
        value = float(corr.loc[feature, EDA_TARGET])
        strength = _correlation_strength(value)
        direction = "positive" if value > 0 else "négative" if value < 0 else "proche de zéro"
        if value > 0:
            interpretation = "les valeurs élevées tendent à être associées à des valeurs élevées de MEDV"
        elif value < 0:
            interpretation = "les valeurs élevées tendent à être associées à des valeurs faibles de MEDV"
        else:
            interpretation = "aucune direction linéaire ne se dégage dans cet échantillon"
        lines.append(
            f"- **{feature} par rapport à MEDV:** {direction} {strength} association "
            f"(Pearson r = {value:.3f}); {interpretation}."
        )

    # Explain outlier flags using the standard 1.5 x IQR screening rule.
    lines.extend(["", "### Valeurs potentiellement atypiques (détection par IQR × 1,5)"])
    for column in numeric_cols:
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((data[column] < lower) | (data[column] > upper)).sum())
        lines.append(
            f"- **{column}:** {count} observation(s) en dehors de "
            f"[{lower:,.2f}, {upper:,.2f}]. Examinez ces enregistrements ; ne les supprimez pas "
            "sans avoir vérifié leur signification."
        )

    lines.extend([
        "",
        "### Règles pratiques d’analyse",
        "1. Vérifiez les valeurs manquantes et les types de données avant d’entraîner les modèles.",
        "2. Examinez les histogrammes et les diagrammes en boîte pour comprendre l’asymétrie et les valeurs inhabituelles.",
        "3. Utilisez la matrice de corrélation pour repérer les associations linéaires, et non des effets causaux.",
        "4. Conservez `MEDV` comme variable cible lorsque l’objectif est d’estimer la valeur d’un logement.",
        "5. Utilisez `RM`, `LSTAT` et `PTRATIO` comme variables explicatives pour l’estimation de la valeur du logement.",
        "6. Séparez les données d’entraînement et de test avant d’ajuster les étapes de prétraitement ou les modèles prédictifs.",
        "7. Comparez les modèles à l’aide de la MAE et de la RMSE (plus faibles = meilleurs) et du R² (plus élevé = généralement meilleur).",
        "8. Examinez la validation croisée et les diagnostics de surapprentissage avant de sélectionner le modèle final.",
        "",
        "**Important :** le jeu de données ne contient que quatre colonnes numériques. Il ne fournit aucune information sur "
        "la localisation, l’âge du bien, le type de propriété ou une devise/unité documentée. "
        "Ne déduisez pas ces caractéristiques à partir des valeurs numériques.",
        "",
        "**Précaution d’interprétation :** la corrélation et l’importance des variables du modèle indiquent "
        "des tendances dans ce jeu de données ; elles ne démontrent pas qu’une variable provoque des changements dans une autre."
    ])
    return "\n".join(lines)


def build_eda_distribution_plots():
    """Affiche un histogramme et un diagramme en boîte marginal pour chaque variable numérique."""
    plots = []
    for column in numeric_cols:
        fig = px.histogram(
            df, x=column, marginal="box", nbins=30,
            title=f"Distribution de {column}",
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
    """Affiche les corrélations de Pearson pour les quatre variables numériques fournies."""
    corr = df[numeric_cols].corr()
    fig = px.imshow(
        corr, text_auto=".2f", zmin=-1, zmax=1,
        color_continuous_scale="RdBu_r",
        title="Matrice de corrélation de Pearson",
        labels={"color": "Corrélation"},
        template="plotly_white"
    )
    fig.update_layout(height=480)
    return pn.pane.Plotly(fig, sizing_mode="stretch_width", height=480)


def build_eda_relationship_plots():
    """Affiche la variable cible en fonction de chacune des variables explicatives fournies."""
    plots = []
    for feature in EDA_FEATURES:
        fig = px.scatter(
            df, x=feature, y=EDA_TARGET,
            title=f"{feature} par rapport à {EDA_TARGET}",
            opacity=0.70, template="plotly_white"
        )
        # Add a simple least-squares trend line without requiring statsmodels.
        slope, intercept = np.polyfit(df[feature].to_numpy(), df[EDA_TARGET].to_numpy(), 1)
        x_line = np.linspace(float(df[feature].min()), float(df[feature].max()), 100)
        fig.add_trace(go.Scatter(
            x=x_line, y=slope * x_line + intercept,
            mode="lines", name="Tendance linéaire"
        ))
        fig.update_layout(height=370)
        plots.append(pn.pane.Plotly(fig, sizing_mode="stretch_width", height=370))
    return pn.Column(
        pn.Row(plots[0], plots[1], sizing_mode="stretch_width"),
        pn.Row(plots[2], sizing_mode="stretch_width"),
        sizing_mode="stretch_width"
    )


def build_eda_summary_table():
    """Renvoie les statistiques descriptives, notamment les quartiles et l’asymétrie."""
    summary = df[numeric_cols].describe().T.reset_index()
    summary = summary.rename(columns={"index": "Variable"})
    summary["skewness"] = df[numeric_cols].skew().values
    return pn.widgets.Tabulator(
        summary.round(3), show_index=False, height=220,
        sizing_mode="stretch_width"
    )


exploratory_analysis_panel = pn.Column(
    pn.pane.Markdown(
        "# Analyse exploratoire des données (AED)\n"
        "Cette section examine le fichier `housing.csv` fourni avant la modélisation. "
        "Les résultats présentés et les règles de détection sont recalculés à partir des données."
    ),
    pn.pane.Markdown(build_exploratory_report()),
    "## Statistiques descriptives",
    build_eda_summary_table(),
    "## Distribution des variables et valeurs potentiellement atypiques",
    build_eda_distribution_plots(),
    "## Relations avec la variable cible (MEDV)",
    build_eda_relationship_plots(),
    "## Corrélation analysis",
    build_eda_correlation_plot(),
    sizing_mode="stretch_width"
)


# ---------------------------------------------------------
# 9. COMPARAISON DES MODÈLES D’APPRENTISSAGE AUTOMATIQUE ET IMPORTANCE DES VARIABLES
# ---------------------------------------------------------
def _build_ml_models():
    """Renvoie un ensemble d’algorithmes de régression pour une comparaison de référence."""
    return {
        "Régression linéaire": LinearRegression(),
        "Régression Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Arbre de décision": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Forêt aléatoire": RandomForestRegressor(
            n_estimators=200, max_depth=None, min_samples_split=2,
            random_state=42, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42
        ),
        "K plus proches voisins": make_pipeline(
            StandardScaler(), KNeighborsRegressor(n_neighbors=5)
        ),
        "Régression par vecteurs de support": make_pipeline(
            StandardScaler(), SVR(kernel="rbf", C=100, epsilon=0.1)
        ),
    }


@pn.depends(target_var_select.param.value)
def run_ml_model_comparison(target_col):
    """Entraîne et compare plusieurs modèles de régression avec une même partition de test."""
    ml_df = df[numeric_cols].dropna()
    X = ml_df.drop(columns=[target_col])
    y = ml_df[target_col]

    if len(ml_df) < 10:
        return pn.pane.Alert(
            "Au moins 10 lignes valides sont nécessaires pour comparer les modèles.",
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
                "Modèle": model_name,
                "R²": r2_score(y_test, pred),
                "MAE": mean_absolute_error(y_test, pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            })
        except Exception as exc:
            rows.append({
                "Modèle": model_name,
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

    best_name = valid_results.iloc[0]["Modèle"]
    best_model = fitted_models[best_name]

    metrics_table = pn.widgets.Tabulator(
        results.round(4), show_index=False, height=300,
        sizing_mode="stretch_width"
    )

    comparison_fig = px.bar(
        valid_results, x="Modèle", y="RMSE", color="Modèle",
        title="Modèle Comparison — Lower RMSE Is Better",
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
        name="Prédiction idéale", line=dict(dash="dash", color="black")
    ))
    pred_fig.update_layout(
        title=f"Valeur réelle par rapport à Valeur prédite Values — {target_col}",
        xaxis_title=f"Valeur réelle {target_col}",
        yaxis_title=f"Valeur prédite {target_col}",
        template="plotly_white"
    )

    # Tree-based models expose feature_importances_; linear models expose coefficients.
    importance = None
    importance_title = None
    if hasattr(best_model, "feature_importances_"):
        importance = best_model.feature_importances_
        importance_title = f"Importance des variables — {best_name}"
    elif hasattr(best_model, "named_steps"):
        final_estimator = list(best_model.named_steps.values())[-1]
        if hasattr(final_estimator, "coef_"):
            importance = np.abs(np.ravel(final_estimator.coef_))
            importance_title = f"Importance des coefficients absolus — {best_name}"
    elif hasattr(best_model, "coef_"):
        importance = np.abs(np.ravel(best_model.coef_))
        importance_title = f"Importance des coefficients absolus — {best_name}"

    components = [
        pn.pane.Markdown(
            f"## Meilleur modèle : **{best_name}**\n"
            f"- **R²:** {valid_results.iloc[0]['R²']:.4f}\n"
            f"- **MAE:** {valid_results.iloc[0]['MAE']:,.4f}\n"
            f"- **RMSE:** {valid_results.iloc[0]['RMSE']:,.4f}\n\n"
            "Modèles use the same 80/20 train-test split (random_state=42)."
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
            "L’importance des variables n’est pas directement disponible pour le meilleur modèle sélectionné."
        ))

    return pn.Column(*components, sizing_mode="stretch_width")


# ---------------------------------------------------------
# 10. PRÉDICTION INTERACTIVE DE LA VALEUR D’UN LOGEMENT
# ---------------------------------------------------------
prediction_model_select = pn.widgets.Select(
    name="Algorithme de prédiction",
    options=[
        "Forêt aléatoire", "Gradient Boosting", "Extra Trees",
        "Arbre de décision", "Régression linéaire"
    ],
    value="Forêt aléatoire",
    width=220,
)
prediction_rm = pn.widgets.FloatInput(
    name="Nombre de pièces (RM)", value=float(df["RM"].median()),
    step=0.1, width=220
)
prediction_lstat = pn.widgets.FloatInput(
    name="Population de statut socio-économique inférieur (LSTAT %)",
    value=float(df["LSTAT"].median()), step=0.1, width=220
)
prediction_ptratio = pn.widgets.FloatInput(
    name="Ratio élèves-enseignant (PTRATIO)",
    value=float(df["PTRATIO"].median()), step=0.1, width=220
)


def predict_house_value(model_name, rm, lstat, ptratio):
    features = df[["RM", "LSTAT", "PTRATIO"]]
    target = df["MEDV"]
    models = {
        "Forêt aléatoire": RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "Arbre de décision": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Régression linéaire": LinearRegression(),
    }
    model = models[model_name]
    model.fit(features, target)
    input_row = pd.DataFrame(
        [[rm, lstat, ptratio]], columns=["RM", "LSTAT", "PTRATIO"]
    )
    predicted_value = float(model.predict(input_row)[0])

    return pn.pane.Markdown(
        f"## Valeur estimée du logement\n"
        f"### ${predicted_value:,.2f}\n"
        f"**Algorithme :** {model_name}\n\n"
        "Il s’agit d’une estimation du modèle fondée sur le jeu de données disponible, et non d’une évaluation immobilière officielle."
    )


prediction_panel = pn.Column(
    "## 🏠 Prédire la valeur d’un logement",
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
# 11. MISE EN PAGE DU TABLEAU DE BORD PANEL
# ---------------------------------------------------------
sidebar = pn.Column(
    "## ⚙️ Paramètres",
    price_tier_select,
    rm_slider,
    pn.layout.Divider(),
    "### 🤖 Paramètres de l’apprentissage automatique et de la recherche sur grille",
    target_var_select,
    k_folds_slider,
    max_depth_slider,
    gs_max_depth_range,
    gs_min_samples_split,
    pn.layout.Divider(),
    "### 🏠 Prédiction de la valeur du logement",
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
    ("🔎 Analyse exploratoire des données", exploratory_analysis_panel),
    ("📊 Analyses statistiques", pn.Column(
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
    ("📈 Régression linéaire et polynomiale", run_linear_vs_polynomial_regression),
    ("🔄 Cross-Validation (K-Pli)", run_kfold_cv),
    ("📉 Diagnostic de complexité et de surapprentissage", get_model_complexity_and_fit_diagnostics),
    ("🔍 Laboratoire de recherche sur grille et résultats", run_grid_search_lab),
    ("🤖 ML Modèle Comparison", run_ml_model_comparison),
    ("🏠 Prédiction de la valeur du logement", prediction_panel)
)

template = pn.template.FastListTemplate(
    title="Analyse des données immobilières – analyse exploratoire, statistiques et apprentissage automatique",
    sidebar=[sidebar],
    main=[tabs],
    accent_base_color="#1f77b4",
    header_background="#1f77b4"
)

template.servable()
