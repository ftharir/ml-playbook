# SMS spam detection

Classify text messages as **spam** or **ham** (legitimate) using TF-IDF features and classical classifiers.

| | |
|---|---|
| **Data** | SMS Spam Collection (Almeida & Hidalgo, UCI Machine Learning Repository): 5,572 English messages, 13.4% spam. |
| **Task** | Binary text classification on imbalanced data |
| **Skills** | NLP basics, TF-IDF, Naive Bayes, linear models, precision/recall trade-offs, error analysis |

## Files

| File | Description |
|---|---|
| `SMS_Spam_Detection(MultinomialNB).ipynb` | Original notebook: TF-IDF + Multinomial Naive Bayes, accuracy 96% |
| `spam.csv` | Dataset |
| `analysis.ipynb` | Follow-up analysis: five models, spam-focused metrics, cross-validated F1, error analysis, most indicative words, a few hand-written test messages |

## Results

The original model has 96% accuracy but only **70% recall on spam**: about 3 in 10 spam messages get through. Accuracy hides this because 87% of messages are ham.

| Model | Accuracy | Precision (spam) | Recall (spam) | F1 (spam) | CV F1 |
|---|---|---|---|---|---|
| Multinomial NB (original approach) | 96.1% | 100% | 70.5% | 0.827 | 0.820 |
| Multinomial NB (tuned) | 98.6% | 99.3% | 89.9% | 0.944 | 0.952 |
| Complement NB | 98.3% | 94.5% | 92.6% | 0.936 | 0.944 |
| Logistic Regression | 98.8% | 98.6% | 92.6% | 0.955 | 0.953 |
| **Linear SVM** | **99.0%** | 98.6% | **94.0%** | **0.962** | **0.957** |

Tuning means bigrams, `min_df=2`, sublinear TF and a smaller smoothing value. The dataset has 403 duplicate messages, so scores are slightly optimistic.

## Run

```bash
pip install -r ../../requirements.txt
jupyter notebook analysis.ipynb
```
