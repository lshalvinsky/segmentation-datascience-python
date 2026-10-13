"""Supervised models that put the segments to work."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42

NUMERIC = [
    "age", "income_k", "tenure_months", "recency_days", "frequency_12m", "monetary_12m",
    "avg_order_value", "discount_usage_rate", "web_sessions_month", "app_sessions_month",
    "email_open_rate", "email_click_rate", "social_engagement", "support_tickets_12m",
    "nps_score", "share_electronics", "share_apparel", "share_home", "share_beauty",
]
CATEGORICAL = ["gender", "region", "preferred_channel"]


def train_segment_classifier(df, segment_col="segment"):
    """Random forest that assigns new customers to a segment and explains the segments.

    Returns the fitted model, held-out classification report and permutation importance.
    """
    X, y = df[NUMERIC + CATEGORICAL], df[segment_col]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    model = Pipeline([
        ("prep", ColumnTransformer([
            ("num", "passthrough", NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ])),
        ("rf", RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=SEED)),
    ]).fit(X_tr, y_tr)
    report = pd.DataFrame(classification_report(y_te, model.predict(X_te), output_dict=True)).T
    imp = permutation_importance(model, X_te, y_te, n_repeats=5, random_state=SEED, n_jobs=-1)
    importance = pd.Series(imp.importances_mean, index=X.columns).sort_values(ascending=False)
    return model, report, importance


def train_response_models(df, target="responded_campaign", segment_col="segment"):
    """Compare a logistic baseline with gradient boosting for campaign response.

    The segment label is included as a feature so the model can learn segment-level
    differences in response on top of individual behaviour.
    """
    features = NUMERIC + CATEGORICAL + [segment_col]
    cats = CATEGORICAL + [segment_col]
    X, y = df[features], df[target]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)

    candidates = {
        "Logistic regression": Pipeline([
            ("prep", ColumnTransformer([
                ("num", StandardScaler(), NUMERIC),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
            ])),
            ("clf", LogisticRegression(max_iter=2000)),
        ]),
        "Gradient boosting": Pipeline([
            ("prep", ColumnTransformer([
                ("num", "passthrough", NUMERIC),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
            ])),
            ("clf", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED)),
        ]),
    }
    results = {}
    for name, model in candidates.items():
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, p)
        results[name] = {"model": model, "auc": roc_auc_score(y_te, p), "fpr": fpr, "tpr": tpr, "scores": p}
    return results, X_te, y_te


def gains_table(y_true, scores, n_bins=10):
    """Decile gains/lift table: target the top-scored customers first."""
    d = pd.DataFrame({"y": np.asarray(y_true), "p": scores}).sort_values("p", ascending=False)
    d["decile"] = np.arange(len(d)) * n_bins // len(d) + 1
    g = d.groupby("decile").agg(customers=("y", "size"), responders=("y", "sum"), avg_score=("p", "mean"))
    g["response_rate"] = g["responders"] / g["customers"]
    g["lift"] = g["response_rate"] / d["y"].mean()
    g["cum_pct_customers"] = g["customers"].cumsum() / len(d)
    g["cum_pct_responders"] = g["responders"].cumsum() / d["y"].sum()
    return g
