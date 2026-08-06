# Heavy Equipment Price Prediction

Predicting the selling price of used heavy equipment (excavators, loaders, motor graders, etc.) sold at auction, based on machine specs, usage and sale details. Built for a Kaggle-style course competition (IIT Madras MLP), scored on RMSLE.

## Approach

1. **EDA** — checked missingness, target skew, categorical cardinality, and correlations between numerical columns and price.
2. **Cleaning** — dropped 15 columns that were >80% empty, and treated `1000`/`1001` in `ManufactureYear` as placeholder values rather than real years.
3. **Feature engineering** — asset age, hours-per-year of usage, sale year/quarter/month, descriptor length, and missingness flags.
4. **Modelling** — compared four models on a held-out validation split:
   - Linear Regression (baseline)
   - Random Forest
   - XGBoost (native categorical support, early stopping)
   - LightGBM (native categorical support, early stopping)
5. **Final submission** — average of LightGBM and XGBoost, each refit on the full training set.

Target is trained in `log1p` space so validation RMSE directly equals the competition's RMSLE.

## Results

| Model             | Validation RMSLE |
|-------------------|-------------------|
| Linear Regression | ~0.50             |
| Random Forest      | ~0.23             |
| XGBoost / LightGBM | best of the four (see notebook for exact numbers) |

Gradient boosting models clearly outperformed the rest — most of the price signal lives in the categorical configuration columns and in engineered features like asset age, not in the raw numeric fields.

## Repo structure

```
.
├── notebook/
│   └── heavy_equipment_price_prediction.ipynb
├── requirements.txt
└── README.md
```

## Replicating this

1. Clone the repo and install dependencies:
   ```bash
   git clone <your-repo-url>
   cd heavy-equipment-price-prediction
   pip install -r requirements.txt
   ```
2. Download the competition data (`train.csv`, `test.csv`, `sample_submission.csv`) from the Kaggle competition page and place it under:
   ```
   ./data/train.csv
   ./data/test.csv
   ./data/sample_submission.csv
   ```
   (or update the `data_dir` path near the top of the notebook to point at wherever you keep it — it currently points at a Kaggle-hosted path).
3. Open the notebook and run top to bottom:
   ```bash
   jupyter notebook notebook/heavy_equipment_price_prediction.ipynb
   ```
4. A `submission.csv` file is written out at the end, ready for upload to the competition.

## Notes

- Random seeds are fixed (`random_state=42`) for reproducibility, but exact metric values can still vary slightly across library versions/hardware.
- The commented-out `RandomizedSearchCV` blocks show the hyperparameter search that was run once beforehand; the best values found are already hard-coded into the model configs so the notebook doesn't need to re-run the full search every time.
