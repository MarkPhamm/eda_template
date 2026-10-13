# EDA Template

A reusable Jupyter notebook for exploratory data analysis (EDA). Point it at a CSV, set the target,
run all cells, and get a structured walkthrough from problem framing to a written summary of findings.

The structure follows the first half of the end-to-end ML checklist (frame the problem → get the data →
clean → explore) and stops where modelling begins.

## Quick start

```bash
uv sync                                   # install dependencies into .venv
uv run jupyter lab notebooks/eda_template.ipynb
```

Without `uv`: `python -m venv .venv && source .venv/bin/activate && pip install pandas numpy matplotlib seaborn scipy statsmodels jupyterlab`.

The notebook runs as-is on the demo dataset (Toyota Corolla used-car prices). To run it headless:

```bash
uv run jupyter nbconvert --to notebook --execute notebooks/eda_template.ipynb --output /tmp/eda_run.ipynb
```

## Use it on your own data

1. Copy your file into `data/raw/`.
2. Edit the **0.2 Config** cell: `DATA_PATH`, `TARGET`, `PROBLEM_TYPE` (`"regression"` / `"classification"`), `ID_COLS`.
3. Run all cells.
4. Fill in the cells marked `✏️ TODO` (problem framing, dataset-specific cleaning, findings).
5. Write up section 7 as `docs/findings_<dataset>.md`.

## Repository layout

```
eda_template/
├── notebooks/
│   └── eda_template.ipynb         # the template (saved without outputs)
├── data/
│   └── raw/toyota_corolla.csv     # demo dataset (1,436 cars × 39 columns)
├── docs/
│   ├── eda_structure.md           # what every section does and what to decide there
│   └── findings_toyota_corolla.md # demo write-up produced with the template
├── reports/
│   └── figures/                   # PNGs saved by the notebook (SAVE_FIGS = True)
├── pyproject.toml
└── uv.lock
```

## Notebook sections

| # | Section | Output |
|---|---|---|
| 0 | Setup | Config, plot style, helper functions |
| 1 | Look at the Big Picture | Problem framing, questions to answer, metric, assumptions |
| 2 | Get the Data | Preview, `info`, `describe`, data dictionary |
| 3 | Data Cleaning (General) | snake_case names, text extraction, dtypes, duplicates, domain checks |
| 4 | Data Quality | Missing-value report, IQR outlier report (detect only) |
| 5 | Attribute Combination | Engineered features |
| 6 | EDA | Column classification → univariate → bivariate (vs target) → multivariate |
| 7 | Summary of Findings | Answers + actions table: the deliverable |
| 8 | Hand-off to Modelling | Next steps (split, preprocess, train, tune, evaluate) |

Details: [`docs/eda_structure.md`](docs/eda_structure.md).

## Demo findings

[`docs/findings_toyota_corolla.md`](docs/findings_toyota_corolla.md) shows the expected output.
Headline: car age explains most of the price (ρ = 0.85); the file is sorted by price, so `id` leaks the target;
equipment flags look predictive but mostly proxy for newer cars.

## Notes

* The notebook is committed **without outputs** to keep diffs readable. Figures used by the docs live in `reports/figures/`.
* Imputation, outlier removal and scaling are **not** done here: they must be fitted on the training split only.
