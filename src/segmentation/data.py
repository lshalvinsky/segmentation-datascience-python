"""Mock customer data for a retail marketing segmentation showcase.

Customers are drawn from five hidden personas with distinct behaviour. The persona
is kept in ``true_persona`` so clustering results can be validated against it, but
it is never used as a model input.
"""

import numpy as np
import pandas as pd

from segmentation import DATA_RAW

REGIONS = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]
CHANNELS = ["store", "web", "app"]
CATEGORIES = ["electronics", "apparel", "home", "beauty"]

# share, age (mean, sd), income $k (median), tenure months (mean), recency scale days,
# orders/yr (mean), avg order value $ (median), discount beta(a, b), web sessions/mo,
# app sessions/mo, email open beta(a, b), social interactions/mo, NPS mean,
# channel probabilities, category Dirichlet alphas
PERSONAS = {
    "Loyal High-Value": {
        "share": 0.15, "age": (48, 9), "income": 95, "tenure": 60, "recency": 15, "freq": 22, "aov": 140,
        "discount": (2, 10), "web": 8, "app": 6, "email": (8, 4), "social": 2, "nps": 8.5,
        "channel": [0.30, 0.50, 0.20], "category": [3, 2, 3, 2],
    },
    "Bargain Hunters": {
        "share": 0.25, "age": (38, 11), "income": 52, "tenure": 30, "recency": 25, "freq": 18, "aov": 35,
        "discount": (8, 3), "web": 12, "app": 4, "email": (5, 5), "social": 3, "nps": 6.5,
        "channel": [0.40, 0.50, 0.10], "category": [2, 4, 3, 3],
    },
    "Digital Natives": {
        "share": 0.22, "age": (26, 5), "income": 48, "tenure": 14, "recency": 30, "freq": 10, "aov": 55,
        "discount": (4, 6), "web": 6, "app": 18, "email": (2, 8), "social": 12, "nps": 7.0,
        "channel": [0.05, 0.25, 0.70], "category": [4, 4, 1, 3],
    },
    "Occasional Big Spenders": {
        "share": 0.13, "age": (45, 10), "income": 120, "tenure": 40, "recency": 90, "freq": 3, "aov": 320,
        "discount": (1, 12), "web": 3, "app": 1, "email": (3, 7), "social": 1, "nps": 7.0,
        "channel": [0.60, 0.35, 0.05], "category": [5, 1, 4, 1],
    },
    "At-Risk Lapsed": {
        "share": 0.25, "age": (41, 13), "income": 60, "tenure": 36, "recency": 220, "freq": 2, "aov": 60,
        "discount": (3, 6), "web": 1, "app": 0.5, "email": (1, 9), "social": 0.5, "nps": 4.5,
        "channel": [0.50, 0.40, 0.10], "category": [2, 3, 3, 2],
    },
}

# Campaign response: base log-odds per persona, before individual behaviour effects
RESPONSE_BASE = {
    "Loyal High-Value": -0.8,
    "Bargain Hunters": -0.6,
    "Digital Natives": -1.4,
    "Occasional Big Spenders": -1.8,
    "At-Risk Lapsed": -3.0,
}


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _counts(rng, mean, n, shape=1.5):
    """Overdispersed (gamma-Poisson) counts - real behaviour is lumpier than Poisson."""
    return rng.poisson(mean * rng.gamma(shape, 1 / shape, n))


def _beta(rng, ab, n, concentration=0.5):
    """Beta draw with the persona's mean but wider spread."""
    a, b = ab
    return rng.beta(a * concentration, b * concentration, n)


def _persona_block(name, p, n, rng):
    age = np.clip(rng.normal(*p["age"], n), 18, 85).round()
    income = rng.lognormal(np.log(p["income"]), 0.4, n).round(1)
    tenure = np.clip(rng.gamma(4, p["tenure"] / 4, n), 1, 180).round()
    recency = np.clip(rng.exponential(p["recency"], n), 0, 730).round()
    frequency = 1 + _counts(rng, max(p["freq"] - 1, 0.1), n)
    aov = rng.lognormal(np.log(p["aov"]), 0.55, n)
    monetary = (frequency * aov * rng.lognormal(0, 0.1, n)).round(2)
    email_open = _beta(rng, p["email"], n)
    categories = rng.dirichlet(p["category"], n)

    df = pd.DataFrame({
        "age": age.astype(int),
        "gender": rng.choice(["F", "M", "Other"], n, p=[0.52, 0.46, 0.02]),
        "region": rng.choice(REGIONS, n),
        "income_k": income,
        "tenure_months": tenure.astype(int),
        "recency_days": recency.astype(int),
        "frequency_12m": frequency,
        "monetary_12m": monetary,
        "avg_order_value": (monetary / frequency).round(2),
        "discount_usage_rate": _beta(rng, p["discount"], n).round(3),
        "web_sessions_month": _counts(rng, p["web"], n),
        "app_sessions_month": _counts(rng, p["app"], n),
        "email_open_rate": email_open.round(3),
        "email_click_rate": (email_open * rng.beta(2, 6, n)).round(3),
        "social_engagement": _counts(rng, p["social"], n),
        "support_tickets_12m": rng.poisson(1.0, n),
        "nps_score": np.clip(rng.normal(p["nps"], 1.8, n), 0, 10).round().astype(int),
        "preferred_channel": rng.choice(CHANNELS, n, p=p["channel"]),
    })
    for i, cat in enumerate(CATEGORIES):
        df[f"share_{cat}"] = categories[:, i].round(3)
    df["true_persona"] = name
    return df


def generate_customers(n=8000, seed=42):
    """Return a DataFrame of ``n`` mock customers with marketing outcome labels."""
    rng = np.random.default_rng(seed)
    names = list(PERSONAS)
    counts = rng.multinomial(n, [PERSONAS[k]["share"] for k in names])
    df = pd.concat(
        [_persona_block(name, PERSONAS[name], c, rng) for name, c in zip(names, counts)],
        ignore_index=True,
    )
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "customer_id", [f"C{i:06d}" for i in range(1, len(df) + 1)])

    # Response to a last-quarter promotional email campaign
    logit = (
        df["true_persona"].map(RESPONSE_BASE)
        + 2.5 * (df["email_open_rate"] - 0.4)
        + 1.2 * (df["discount_usage_rate"] - 0.4)
        - 0.004 * (df["recency_days"] - 60)
        + 0.15 * (df["nps_score"] - 6)
        + rng.normal(0, 0.5, len(df))
    )
    df["responded_campaign"] = rng.binomial(1, _sigmoid(logit))

    # Churn in the following 6 months
    churn_logit = (
        -2.2
        + 0.012 * (df["recency_days"] - 60)
        - 0.35 * (df["nps_score"] - 6)
        + 0.25 * df["support_tickets_12m"]
        - 0.02 * df["tenure_months"]
        + rng.normal(0, 0.5, len(df))
    )
    df["churned_6m"] = rng.binomial(1, _sigmoid(churn_logit))
    return df


def save_customers(df, path=None):
    path = path or DATA_RAW / "customers.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_customers(path=None):
    path = path or DATA_RAW / "customers.csv"
    if not path.exists():
        save_customers(generate_customers(), path)
    return pd.read_csv(path)
