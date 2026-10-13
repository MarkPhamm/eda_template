# EDA Template Structure

What each section of [`notebooks/eda_template.ipynb`](../notebooks/eda_template.ipynb) is for, what it outputs,
and what you need to decide there.

The template mirrors the first half of the end-to-end ML checklist (frame → get data → clean → explore)
and stops before the train/test split. Its deliverable is section 7, **Summary of Findings**.

```
0. Setup                 imports, config, plot style, helpers
1. Big Picture           problem, questions, metric, assumptions        ✏️ written by you
2. Get the Data          load, peek, data dictionary
3. Data Cleaning         names, text fields, dtypes, duplicates, typos  ✏️ dataset-specific
4. Data Quality          missing values, outliers (detect only)
5. Attribute Combination engineered features                             ✏️ dataset-specific
6. EDA                   classify → univariate → bivariate → multivariate
7. Summary of Findings   the deliverable                                 ✏️ written by you
8. Hand-off              next steps for the modelling notebook
```

`✏️ TODO` cells are the only places you must edit. Everything else adapts to the columns it finds.

---

## 0. Setup

| Cell | Purpose |
|---|---|
| 0.1 Imports | pandas, numpy, matplotlib, seaborn, scipy. |
| **0.2 Config** | **The only cell to change for a new dataset**: `DATA_PATH`, `TARGET`, `PROBLEM_TYPE`, `ID_COLS`, thresholds. |
| 0.3 Plot style | One palette for all charts: blue for single series, fixed order for hue splits (max 3 groups), blue ↔ red diverging for correlations. |
| 0.4 Helpers | `to_snake_case`, `plot_grid`, `missing_report`, `outlier_report`, `classify_columns`, `save_fig`. |

**Config reference**

| Setting | Default | Meaning |
|---|---|---|
| `TARGET` | `"price"` | Target column (snake_case). `None` skips target-based sections. |
| `PROBLEM_TYPE` | `"regression"` | `"classification"` switches target plots to class balance and stacked bars. |
| `ID_COLS` | `["id"]` | Excluded from analysis; checked for leakage in 6.3. |
| `CAT_COLS` | `None` | Force columns to categorical (e.g. numeric zip codes). |
| `CAT_MAX_UNIQUE` | `15` | Numeric columns with ≤ this many values are treated as **discrete**. |
| `HIGH_CORR` | `0.8` | Pairs above this \|ρ\| are flagged as redundant. |
| `SAVE_FIGS` | `True` | Write every figure to `reports/figures/` for use in docs. |

## 1. Look at the Big Picture

Written before looking at data, so the EDA answers specific questions instead of producing every plot possible.

* **1.1 Frame the problem**: business objective, how the output is used, target type.
* **1.2 Questions & hypotheses**: a table of questions, each mapped to a later section. Section 7 answers these.
* **1.3 Performance measure**: chosen now because it changes what to check (skewed target → RMSLE; imbalanced classes → PR-AUC).
* **1.4 Assumptions**: units, meaning of rows, how the target was recorded. Confirm with the data owner.

## 2. Get the Data

`head`, `info`, `describe(include="all")`, and an auto-generated **data dictionary** (dtype, unique count,
example value, empty `meaning` column to fill in).

## 3. Data Cleaning (General)

Cleaning that is safe on the full dataset because it uses no target or test-set statistics.

| Step | What it does |
|---|---|
| 3.1 Column names | snake_case everything; manual renames. |
| 3.2 Data body | Strip whitespace; extract structured fields from free text (demo: `body_type`, `trim` from `model`). |
| 3.3 Datatypes | Dates → datetime, flags → int, ordered labels → ordered categorical. |
| 3.4 Duplicates | Exact duplicates **and** duplicates ignoring ID columns (the more common case). |
| 3.5 Domain sanity checks | Values that are storable but impossible (demo: `cc = 16000`). These are errors, not outliers. |

## 4. Data Quality

**Detect and decide only.** Imputers and outlier caps are fitted after the train/test split, in the modelling notebook.

* **4.1 Missing values**: count and % per column, bar chart, null co-occurrence matrix, null-target check.
  Reference notes on MCAR / MAR / MNAR / by-design and the four handling options.
* **4.2 Outliers**: IQR table (bounds, count, %) and box plots for every numeric column.
  For each flagged column decide: real but rare (keep, maybe log) or error (fix in 3.5).

## 5. Attribute Combination

Ratios, rates, flags and date parts that might explain the target better than raw columns.
They go through section 6 like every other column, so weak ones get caught (demo: `hp_per_tonne` turned out useless).

## 6. EDA

### 6.0 Classify attributes

`classify_columns` puts every column in exactly one group. All later plots are driven by these groups.

| Group | Rule | Plotted as |
|---|---|---|
| constant | 1 unique value | dropped |
| binary | 2 unique values | count bars; target lift table |
| categorical | non-numeric, or listed in `CAT_COLS` | top-N bars; box plot vs target |
| discrete | numeric, ≤ `CAT_MAX_UNIQUE` values | count bars; box plot vs target |
| continuous | numeric, more values | histogram + KDE; scatter vs target |

Categoricals with more than 50 levels are summarised as tables instead of plotted.

### 6.1 Univariate: one column at a time

| Cell | Output | Look for |
|---|---|---|
| 6.1.1 Target | Histogram raw vs `log1p`, percentiles (regression) or class balance (classification) | Skew, multi-modality, imbalance |
| 6.1.2 Numeric | Skew table; histograms; count bars | Skew > 1 → transform candidates |
| 6.1.3 Binary | % majority value; count bars | > 95% one value → near-constant |
| 6.1.4 Categorical | Top-N bars; rare-level list (< 1%) | Levels to fold into "other" |

### 6.2 Bivariate: each feature against the target

| Cell | Output | Look for |
|---|---|---|
| Correlation ranking | Spearman ρ of every numeric/binary feature with target, sorted | Candidate drivers |
| Scatter + LOWESS | Top 8 continuous features vs target with a smoothed trend (classification: KDE per class) | Non-linear shapes, thresholds, class separation |
| Category box plots | Target distribution per level (classification: stacked class rates, binary flags included) | Ordered effects, big gaps between levels |
| Binary lift table | Median target for each of the two values, Mann-Whitney p-value (regression only) | Flags that shift the target |

Spearman (rank) correlation is used throughout because it captures monotonic non-linear relationships and is robust to outliers.

### 6.3 Multivariate: features against each other

| Cell | Output | Look for |
|---|---|---|
| Correlation heatmap | Lower-triangle Spearman matrix | Clusters of related features |
| High-correlation pairs | Pairs with \|ρ\| ≥ `HIGH_CORR` | Redundant features: keep one |
| Leakage check | ρ between ID columns and target | Sorted files, IDs that encode the target |
| 6.3.1 Segments | Target stats and scatter plots split by `HUE` (max 3 groups + "other") | Interactions: relationships that change by segment |
| Pairplot | Top 4 continuous features + target | Joint structure, clusters |

## 7. Summary of Findings

The deliverable. Answer every question from 1.2 in one line, then an **actions table**
(action · columns · why) that the modelling notebook implements. Copy it into `docs/` as a findings doc.
Example: [`findings_toyota_corolla.md`](findings_toyota_corolla.md).

## 8. Hand-off to Modelling

Pointer to the rest of the end-to-end checklist: split → clean for modelling → preprocess pipeline →
train → tune → explain → test-set evaluation. Optionally saves the cleaned frame to `data/processed/`.

---

## Adapting to a new dataset

1. Put the file in `data/raw/`.
2. Edit **0.2 Config**.
3. Run all. Fix anything that errors in 3.2 / 3.5 / 5 (the dataset-specific demo cells skip themselves when their columns are missing).
4. Fill in the `✏️ TODO` cells: 1.x, 3.2, 3.5, 5, 6.3.1 `HUE`, 7.
5. Write `docs/findings_<dataset>.md` from section 7.
