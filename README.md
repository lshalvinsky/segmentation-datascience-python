# segmentation-datascience-ltcs

Segmentation data science project.

## Layout

```
data/raw/          original, immutable data (not committed)
data/processed/    cleaned / feature data (not committed)
notebooks/         exploratory Jupyter notebooks
src/segmentation/  reusable project code
tests/             pytest tests
reports/figures/   generated plots
```

## Setup (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
python -m ipykernel install --user --name segmentation-ltcs
```

Then launch `jupyter lab`, or open the folder in VS Code and select the `.venv` interpreter.
