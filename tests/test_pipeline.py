from segmentation.clustering import fit_kmeans, index_vs_average, name_segments
from segmentation.data import PERSONAS, generate_customers
from segmentation.features import (
    CLUSTER_FEATURES,
    PROFILE_FEATURES,
    cluster_preprocessor,
    rfm_scores,
)


def test_generate_customers_shape_and_labels():
    df = generate_customers(n=500, seed=1)
    assert len(df) == 500
    assert df["customer_id"].is_unique
    assert set(df["true_persona"]) <= set(PERSONAS)
    assert df["responded_campaign"].isin([0, 1]).all()
    assert (df["avg_order_value"] > 0).all()


def test_rfm_scores_in_range():
    rfm = rfm_scores(generate_customers(n=500, seed=1))
    for col in "RFM":
        assert rfm[col].between(1, 5).all()


def test_kmeans_names_are_unique():
    df = generate_customers(n=1500, seed=2)
    X = cluster_preprocessor().fit_transform(df[CLUSTER_FEATURES])
    df["cluster"] = fit_kmeans(X, 5).labels_
    names = name_segments(index_vs_average(df, "cluster", PROFILE_FEATURES))
    assert len(set(names.values())) == 5
