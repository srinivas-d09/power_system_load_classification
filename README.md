# Assignment 2 — Power System Load Type Classification

## Objective
Build a supervised machine-learning classifier to predict one of three power-system load categories from historical readings:

- `Light_Load`
- `Medium_Load`
- `Maximum_Load`

The evaluation follows the assignment requirement: the **latest calendar month is reserved as the test set**. Earlier months are used for training.

## Project structure
```text
Assignment_2_Power_System_Load_Classification/
├── app.py
├── train_model.py
├── load_type_classification.ipynb
├── requirements.txt
├── Problem_Statement.docx
├── README.md
├── data/
│   └── README.txt
├── figures/       # generated when the model is run
├── artifacts/     # model, metrics, and predictions generated at runtime
└── extras/
    ├── dashboard_prototype.html
    └── train_with_visualizations_alternative.py
```

## Important dataset note
The original CSV dataset was **not included in the files available for packaging**. It has not been invented. Before running the notebook or training script, place the original assignment CSV at `data/load_data.csv`. The file should contain a date/time column, predictor columns, and the `Load_Type` target.

## Setup (Windows PowerShell)
Open PowerShell in this project folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If virtual-environment activation is blocked, run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the notebook
After adding `data/load_data.csv`:

```powershell
jupyter notebook load_type_classification.ipynb
```

Run the notebook cells from top to bottom.

## Train and evaluate the model
```powershell
python train_model.py
```

To use a CSV stored at another location:

```powershell
python train_model.py --data "C:\path\to\load_data.csv"
```

The training script uses an Extra Trees classifier by default and saves:
- `figures/` — class distribution, time-series, correlation, feature importance, confusion matrices, and evaluation charts
- `artifacts/load_type_model.joblib` — trained model bundle
- `artifacts/metrics.json` — evaluation metrics
- `artifacts/last_month_predictions.csv` — actual and predicted labels for the holdout month

Metrics include accuracy, balanced accuracy, macro precision/recall/F1, and per-class precision/recall/F1. Actual metrics are only available after training on the real dataset.

## Launch the interactive dashboard
```powershell
streamlit run app.py
```

The Streamlit dashboard supports CSV upload for exploration. Model metrics and single-record predictions appear after the dataset is added and the training script has generated the artifacts.

## Methodology
1. Validate the date/time and target columns.
2. Sort observations chronologically.
3. Create calendar and cyclical time features.
4. Hold out the latest calendar month for testing.
5. Fit missing-value medians on training data only.
6. Train the classifier and evaluate it on the untouched holdout month.
7. Save evaluation charts, metrics, predictions, and the trained model.

## Submission checklist
- [ ] Add the original assignment CSV to `data/load_data.csv`.
- [ ] Install the dependencies.
- [ ] Run all notebook cells.
- [ ] Run `python train_model.py`.
- [ ] Confirm that `figures/` and `artifacts/` contain generated outputs.
- [ ] Launch the dashboard and check it.
- [ ] Submit the ZIP only after including the real dataset and, if required, generated outputs.

## Notes
The `extras/dashboard_prototype.html` file is a standalone visual prototype. Its displayed example figures are static and should **not** be treated as measured results from this packaged model. The primary interactive dashboard is `app.py`.

`extras/train_with_visualizations_alternative.py` is an additional alternative training script. Use `train_model.py` as the main training entry point.
