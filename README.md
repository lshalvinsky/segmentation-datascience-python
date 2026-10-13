# segmentation-datascience-ltcs

End-to-end **marketing customer segmentation** showcase on mock retail data: behavioural
clustering, segment profiling, explainability, campaign-response modelling and a targeting playbook.

**Start here:** [notebooks/01_marketing_segmentation.ipynb](notebooks/01_marketing_segmentation.ipynb)
(committed with outputs, so it renders on GitHub without running anything).

## What's inside

| Step | Method |
|---|---|
| Mock data | 8,000 customers drawn from 5 hidden personas: RFM, channel engagement, discounts, email, NPS, category mix, campaign response, churn |
| Baseline | Rule-based RFM quintile segmentation |
| Choosing *k* | Elbow, silhouette, bootstrap stability (ARI), Gaussian-mixture BIC |
| Clustering | K-Means vs Gaussian Mixture vs Ward hierarchical, validated against the hidden personas |
| Profiling | z-score index heatmap, PCA segment map, archetype naming via Hungarian matching |
| Explainability | Random forest segment classifier + permutation importance |
| Targeting | Logistic regression vs gradient boosting response models, ROC, decile lift / cumulative gains |
| Action | Per-segment playbook: size, revenue share, response, churn, recommended action |

## Headline results

- K-Means with k = 5 recovers the hidden personas with ARI 0.86. The random forest reassigns segments with macro F1 0.95.
- *Loyal High-Value* customers are 15% of customers but bring in 52% of revenue.
- RFM mislabels 61% of *Occasional Big Spenders* as at-risk or hibernating.
- Targeting the top 30% by predicted response captures 65% of responders.

## Layout

```
data/raw/          generated customers.csv (not committed - regenerate below)
data/processed/    customer_segments.csv output (not committed)
notebooks/         analysis notebook
scripts/           generate_data.py
src/segmentation/  data.py, features.py, clustering.py, models.py, plotting.py
tests/             pytest tests
```

## Setup (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
python -m ipykernel install --user --name segmentation-ltcs --display-name "Python (segmentation-ltcs)"

python scripts/generate_data.py      # optional: the notebook generates data if missing
jupyter lab                          # or open the folder in VS Code
pytest
```
