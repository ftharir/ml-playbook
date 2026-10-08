# Heart disease risk prediction

Predict whether a patient will develop **coronary heart disease within 10 years** (`TenYearCHD`) from demographic, behavioural and medical features.

| | |
|---|---|
| **Data** | `framingham.csv`: 4,240 patients, 15 features, from the Framingham Heart Study (as distributed on Kaggle). 15.2% positive. |
| **Task** | Binary classification on an imbalanced medical dataset |
| **Skills** | Data cleaning, imbalanced-class evaluation, ROC analysis, interpretable models |

## Files

| File | Description |
|---|---|
| `Heart_Disease_Prediction.ipynb` | Original notebook: logistic regression with standardised features, training accuracy 86.0% |
| `framingham.csv` | Dataset |
| `analysis.ipynb` | Follow-up analysis: median imputation inside a pipeline, four models, ROC curves, confusion matrix, cross-validated AUC, coefficient interpretation |

## Results

Stratified 80/20 split; cross-validation is 5-fold ROC-AUC on all data.

| Model | Accuracy | ROC-AUC (test) | Recall (CHD) | Precision (CHD) | CV ROC-AUC |
|---|---|---|---|---|---|
| Always predict "no CHD" | 84.8% | 0.500 | 0% | n/a | n/a |
| **Logistic Regression** | 84.4% | 0.702 | 5.4% | 41.2% | **0.723** |
| Logistic Regression (balanced) | 67.1% | 0.700 | **60.5%** | 25.5% | 0.723 |
| Random Forest | 84.6% | 0.664 | 2.3% | 37.5% | 0.711 |
| Gradient Boosting | 84.2% | 0.660 | 8.5% | 40.7% | 0.699 |

Main takeaways:
- Accuracy is misleading here: a model that never predicts disease already scores 84.8%, so the original 86.0% training accuracy is barely above that baseline.
- A plain logistic regression is as good as the tree ensembles, and it is interpretable. The strongest risk factors are **age, systolic blood pressure, cigarettes per day, sex and glucose**.
- The decision threshold is a real trade-off: balancing the classes finds 60% of at-risk patients but flags many more false alarms.

## Run

```bash
pip install -r ../../requirements.txt
jupyter notebook analysis.ipynb
```

To re-run the original notebook as well, install `requirements-original.txt` (it imports `statsmodels`).

> Not medical advice. This is a learning project.
