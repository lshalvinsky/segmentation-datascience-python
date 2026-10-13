"""Clustering, model selection and segment profiling."""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

SEED = 42


def evaluate_k(X, k_range=range(2, 11), sample_size=3000):
    """Inertia and internal validity scores for K-Means over a range of k."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X, km.labels_, sample_size=sample_size, random_state=SEED),
            "calinski_harabasz": calinski_harabasz_score(X, km.labels_),
            "davies_bouldin": davies_bouldin_score(X, km.labels_),
        })
    return pd.DataFrame(rows).set_index("k")


def stability_by_k(X, k_range=range(2, 11), n_boot=10, frac=0.8):
    """Mean pairwise ARI of K-Means fits on bootstrap subsamples (higher = more reproducible).

    Each fit predicts labels for the full dataset so runs can be compared directly.
    """
    rng = np.random.default_rng(SEED)
    out = {}
    for k in k_range:
        runs = []
        for b in range(n_boot):
            idx = rng.choice(len(X), int(frac * len(X)), replace=False)
            runs.append(KMeans(n_clusters=k, n_init=5, random_state=b).fit(X[idx]).predict(X))
        out[k] = np.mean([adjusted_rand_score(runs[i], runs[j])
                          for i in range(n_boot) for j in range(i + 1, n_boot)])
    return pd.Series(out, name="stability")


def gmm_bic(X, k_range=range(2, 11)):
    """BIC for full-covariance Gaussian mixtures (lower is better)."""
    return pd.Series(
        {k: GaussianMixture(k, covariance_type="full", random_state=SEED).fit(X).bic(X) for k in k_range},
        name="bic",
    )


def fit_kmeans(X, k):
    return KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(X)


def compare_algorithms(X, k, truth=None, sample_size=3000):
    """Fit K-Means, GMM and Ward hierarchical clustering with the same k and score them."""
    labels = {
        "K-Means": fit_kmeans(X, k).labels_,
        "Gaussian Mixture": GaussianMixture(k, covariance_type="full", random_state=SEED).fit_predict(X),
        "Agglomerative (Ward)": AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X),
    }
    rows = []
    for name, lab in labels.items():
        row = {
            "algorithm": name,
            "silhouette": silhouette_score(X, lab, sample_size=sample_size, random_state=SEED),
            "calinski_harabasz": calinski_harabasz_score(X, lab),
            "davies_bouldin": davies_bouldin_score(X, lab),
        }
        if truth is not None:
            row["ARI_vs_hidden_persona"] = adjusted_rand_score(truth, lab)
        rows.append(row)
    return pd.DataFrame(rows).set_index("algorithm"), labels


def profile_segments(df, segment_col, features):
    """Mean of each feature per segment, plus segment size and share."""
    prof = df.groupby(segment_col)[features].mean()
    prof.insert(0, "share", df[segment_col].value_counts(normalize=True))
    prof.insert(0, "customers", df[segment_col].value_counts())
    return prof


def index_vs_average(df, segment_col, features):
    """Segment mean expressed as a z-score against the whole customer base."""
    mu, sd = df[features].mean(), df[features].std()
    return (df.groupby(segment_col)[features].mean() - mu) / sd


# Business archetypes expressed as the direction we expect each feature to deviate
# from the average. Clusters are named by best one-to-one match to these hypotheses.
ARCHETYPES = {
    "Loyal High-Value": {"frequency_12m": 1, "monetary_12m": 1, "recency_days": -1, "email_open_rate": 1, "nps_score": 1},
    "Bargain Hunters": {"discount_usage_rate": 2, "avg_order_value": -1, "web_sessions_month": 1},
    "Digital Natives": {"app_sessions_month": 2, "social_engagement": 1, "age": -1},
    "Occasional Big Spenders": {"avg_order_value": 2, "frequency_12m": -1, "income_k": 1},
    "At-Risk Lapsed": {"recency_days": 2, "web_sessions_month": -1, "email_open_rate": -1, "nps_score": -1},
}


def name_segments(z_profile):
    """Map cluster id -> archetype name via optimal assignment on archetype scores."""
    names = list(ARCHETYPES)
    clusters = list(z_profile.index)
    score = np.array([
        [sum(w * z_profile.loc[c, f] for f, w in ARCHETYPES[n].items()) for n in names]
        for c in clusters
    ])
    rows, cols = linear_sum_assignment(-score)
    mapping = {c: f"Segment {c}" for c in clusters}
    mapping.update({clusters[r]: names[col] for r, col in zip(rows, cols)})
    return mapping
