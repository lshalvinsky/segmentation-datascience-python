"""Feature engineering: clustering inputs and classic RFM scoring."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# Behavioural features used to form segments. Demographics are deliberately left
# out so segments describe what customers *do*; demographics are used to profile.
CLUSTER_FEATURES = [
    "recency_days",
    "frequency_12m",
    "avg_order_value",
    "discount_usage_rate",
    "web_sessions_month",
    "app_sessions_month",
    "email_open_rate",
    "social_engagement",
]
SKEWED = [
    "recency_days",
    "frequency_12m",
    "avg_order_value",
    "web_sessions_month",
    "app_sessions_month",
    "social_engagement",
]

PROFILE_FEATURES = CLUSTER_FEATURES + [
    "monetary_12m",
    "tenure_months",
    "age",
    "income_k",
    "nps_score",
    "email_click_rate",
    "share_electronics",
    "share_apparel",
    "share_home",
    "share_beauty",
    "responded_campaign",
    "churned_6m",
]


def cluster_preprocessor():
    """log1p the skewed columns, then standardise everything."""
    unskewed = [c for c in CLUSTER_FEATURES if c not in SKEWED]
    return Pipeline([
        ("transform", ColumnTransformer([
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"), SKEWED),
            ("pass", "passthrough", unskewed),
        ])),
        ("scale", StandardScaler()),
    ])


def rfm_scores(df):
    """Quintile R, F, M scores (5 = best) and the standard rule-based RFM segment."""
    out = pd.DataFrame(index=df.index)
    # rank first so ties don't break qcut's bin edges
    out["R"] = pd.qcut(df["recency_days"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    out["F"] = pd.qcut(df["frequency_12m"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    out["M"] = pd.qcut(df["monetary_12m"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    out["rfm_segment"] = [_rfm_label(r, f, m) for r, f, m in zip(out["R"], out["F"], out["M"])]
    return out


def _rfm_label(r, f, m):
    fm = (f + m) / 2
    if r >= 4 and fm >= 4:
        return "Champions"
    if r >= 3 and fm >= 3:
        return "Loyal"
    if r >= 4:
        return "Promising"
    if r <= 2 and fm >= 3:
        return "At Risk"
    if r <= 2:
        return "Hibernating"
    return "Needs Attention"
