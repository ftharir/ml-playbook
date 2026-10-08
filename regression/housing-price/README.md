# Housing price regression

Predict the **median house value of a California district** from its location, age, size and median income.

| | |
|---|---|
| **Data** | California housing data (1990 census, Pace & Barry), as distributed on Kaggle: 20,640 districts, 9 numeric features plus `ocean_proximity`. |
| **Task** | Regression |
| **Skills** | Feature engineering, scikit-learn pipelines, tree ensembles, cross-validation, residual analysis |

## Files

| File | Description |
|---|---|
| `HousePricePrediction (1).ipynb` | Original notebook: log transform, one-hot encoding, two engineered ratios, Linear Regression and a grid-searched Random Forest |
| `housing.csv` | Dataset |
| `archive.zip` | Archive the dataset came from |
| `analysis.ipynb` | Follow-up analysis: same preprocessing ideas rebuilt as a leak-free pipeline with a fixed seed; four models, RMSE/MAE/R², cross-validation, residual and per-region error analysis, feature importance |

## Results

80/20 split with a fixed seed; cross-validation is 5-fold R² on all data.

| Model | RMSE | MAE | R² (test) | R² (5-fold CV) |
|---|---|---|---|---|
| Linear Regression | $67.3k | $48.4k | 0.655 | 0.670 |
| Ridge | $67.3k | $48.4k | 0.655 | 0.670 |
| Random Forest | $49.1k | $31.8k | 0.816 | 0.822 |
| **Gradient Boosting** | **$46.9k** | **$31.4k** | **0.832** | **0.841** |

Original notebook for reference: Linear Regression R² 0.676, tuned Random Forest R² 0.822 (no random seed, R² only).

Main takeaways:
- Tree models beat linear ones by about 15 R² points, so the price–location relationship is strongly non-linear.
- Errors are lowest inland (MAE ≈ $23.5k) and highest on the coast and bay (≈ $38k).
- The target is capped at $500,001 (4.7% of districts), which limits accuracy at the top end.

## Run

```bash
pip install -r ../../requirements.txt
jupyter notebook analysis.ipynb
```

Takes about 3 minutes (cross-validation of the tree models).
