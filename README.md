# ML Playbook

Five end-to-end machine learning case studies covering **classification, regression, NLP and computer vision**. Each project contains the notebook I first wrote while learning, plus a follow-up analysis with a more rigorous evaluation.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-F7931E?logo=scikitlearn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

## About these projects

The notebooks were first written in **February 2024**, when I was learning machine learning, and I am publishing them now. They are the notebooks I wrote while learning. The follow-up work was added in 2026 as separate files next to them:

- `analysis.ipynb`: the same problem re-evaluated with better methodology and metrics,
- a README per project with results,
- shared setup files (`requirements.txt`, `.gitignore`).

In each project folder, the notebook named after the task is the original; `analysis.ipynb` is the follow-up.

## Projects

| Area | Project | Original result | Follow-up result |
|---|---|---|---|
| Classification | [Breast cancer](classification/breast-cancer) | 94.1% accuracy (Naive Bayes) | **97.9%** accuracy; malignant recall 91% → **98.5%** (Logistic Regression) |
| Classification | [Heart disease risk](classification/heart-disease) | Logistic regression, 86% training accuracy only | Held-out evaluation; CV ROC-AUC **0.723**; CHD recall 5% → 60% with class balancing |
| Classification (NLP) | [SMS spam detection](classification/sms-spam) | 96% accuracy, but only 70% spam recall | Spam recall **94.0%**, F1 **0.962** (Linear SVM) |
| Regression | [Housing prices](regression/housing-price) | R² 0.822 (tuned Random Forest) | R² **0.832**, RMSE ≈ $46.9k (Gradient Boosting), with CV and error analysis |
| Deep learning | [Hot dog or not (CNN)](deep-learning/hotdog-cnn) | Training log inside the notebook | Training curves plotted from the log; 73% validation accuracy at epoch 18 of 50 |

## What the follow-up analyses add

- **Metrics that match the problem**: recall for malignant tumours, spam and heart-disease cases instead of accuracy alone; RMSE and MAE next to R².
- **Reliable comparison**: several models per problem, fixed random seeds, and cross-validation instead of a single train/test split.
- **Leak-free preprocessing**: imputation, scaling and encoding inside scikit-learn pipelines fitted on training data only.
- **Error analysis**: confusion matrices, residuals, per-region errors, misclassified messages.
- **Interpretation**: feature importances and model coefficients.
- **Honest reporting**: caveats on every project (small samples, capped targets, duplicated rows).

## Skills demonstrated

- **Classical ML (scikit-learn)**: Logistic Regression, SVM, k-NN, Naive Bayes, Random Forest, Gradient Boosting, Ridge; pipelines, `ColumnTransformer`, grid search
- **Evaluation**: stratified splits, cross-validation, ROC-AUC, PR-AUC, precision/recall/F1, confusion matrices, residual analysis, handling imbalanced classes
- **Data work (pandas, NumPy)**: cleaning, missing values, log transforms, one-hot encoding, feature engineering
- **NLP**: TF-IDF with n-grams, Naive Bayes and linear text classifiers
- **Deep learning (TensorFlow / Keras)**: CNN design, data augmentation, `tf.data` pipelines, class balancing
- **Visualisation**: matplotlib and seaborn
- **Reproducibility**: fixed seeds, dependency files, notebooks that ship with their outputs

## Repository structure

```
.
├── classification/
│   ├── breast-cancer/
│   ├── heart-disease/
│   └── sms-spam/
├── regression/
│   └── housing-price/
├── deep-learning/
│   └── hotdog-cnn/
├── requirements.txt            for the follow-up analyses
├── requirements-original.txt   extra packages to re-run the original notebooks
└── README.md
```

Each project folder contains the original notebook, its dataset (if any), `analysis.ipynb` (or a plotting script for the CNN) and a README.

## Getting started

```bash
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
jupyter notebook
```

Open any `analysis.ipynb`. They already contain their outputs, so they can also be read directly on GitHub. To re-run the original notebooks use `requirements-original.txt` instead (TensorFlow is heavy, and the CNN project downloads a 4.65 GiB dataset).

## Data sources

| Project | Dataset | Source |
|---|---|---|
| Breast cancer | Breast Cancer Wisconsin (Diagnostic) | Bundled with scikit-learn; UCI Machine Learning Repository |
| Heart disease | Framingham Heart Study extract (`framingham.csv`) | Kaggle distribution of the Framingham Heart Study data |
| SMS spam | SMS Spam Collection | Almeida & Hidalgo, UCI Machine Learning Repository |
| Housing | California housing (1990 census) | Pace & Barry (1997), as distributed on Kaggle |
| Hot dog CNN | Food-101 | Bossard et al. (2014), downloaded via TensorFlow Datasets, not included |

The datasets belong to their respective owners and are included only for reproducibility of these educational projects. Please check each source's terms before reusing them.

## Future work

- A **model card** per project and calibration plots for the medical examples
- **Hyperparameter search** with Optuna and nested cross-validation
- **Transfer learning** (MobileNetV2 / EfficientNet) for the hot-dog classifier, with a confusion matrix and error gallery
- **Explainability** with SHAP for the heart-disease and housing models
- **Transformer-based** text classification for the SMS task, compared with the TF-IDF baselines
- A small **Streamlit** app to try the spam classifier and house-price model interactively
- **Tests and CI**: smoke-run the notebooks with `nbmake` in GitHub Actions
