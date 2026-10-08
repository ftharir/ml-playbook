# Breast cancer classification

Classify breast tumours as **malignant** or **benign** from 30 numeric features computed from cell-nucleus images.

| | |
|---|---|
| **Data** | Breast Cancer Wisconsin (Diagnostic), loaded from `sklearn.datasets.load_breast_cancer`. 569 samples, 212 malignant / 357 benign. Original source: UCI Machine Learning Repository (Wolberg, Street, Mangasarian). |
| **Task** | Binary classification |
| **Skills** | scikit-learn, model comparison, cross-validation, recall-oriented evaluation |

## Files

| File | Description |
|---|---|
| `Cancer_cell_classification.ipynb` | Original notebook: Gaussian Naive Bayes, accuracy 94.1% |
| `analysis.ipynb` | Follow-up analysis: 5-model comparison, cross-validation, confusion matrix, feature importance |

## Results

Same train/test split as the original notebook (`test_size=0.33`, `random_state=42`); *malignant* is the positive class.

| Model | Accuracy | Recall (malignant) | Missed malignant | 5-fold CV accuracy |
|---|---|---|---|---|
| Gaussian Naive Bayes (original) | 94.1% | 91.0% | 6 of 67 | 93.9% |
| k-NN | 95.7% | 94.0% | 4 | 96.3% |
| Random Forest | 95.7% | 92.5% | 5 | 95.3% |
| SVM (RBF) | 96.8% | 97.0% | 2 | **97.7%** |
| **Logistic Regression** | **97.9%** | **98.5%** | **1** | 97.4% |

Logistic Regression and SVM are statistically tied in cross-validation; both clearly beat Naive Bayes. In a screening setting, missed malignant cases (false negatives) are the costly error, which is why recall is reported next to accuracy.

## Run

```bash
pip install -r ../../requirements.txt
jupyter notebook analysis.ipynb
```

The notebooks already contain their outputs, so they can be read on GitHub without running them.
