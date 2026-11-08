# Phishing Website Detection – Model Comparison Report

## 1. Dataset

- Dataset file: `dataset/upload.csv`
- Total records: 11,055
- Input features: 30
- Target: `Result`
- Label convention: `-1` = phishing; `1` = legitimate
- Class counts: 4,898 phishing and 6,157 legitimate
- Missing values: 0
- Duplicate rows: 0

`Id` was excluded because it is a record identifier, not a predictive feature. The target column `Result` was separated from the 30 input features.

## 2. Experimental protocol

- Train/test split: 80% / 20%
- Stratification: enabled
- Random state: 42
- Training records: 8,844
- Test records: 2,211
- Cross-validation: stratified 5-fold CV on the training partition only
- Phishing (`-1`) was explicitly treated as the positive class when calculating phishing precision, recall and F1.
- The test partition was not used for cross-validation or model selection.

Logistic Regression was placed in a StandardScaler pipeline. Random Forest and Gradient Boosting were trained on the original numeric feature values.

## 3. Algorithms

1. Logistic Regression – linear baseline classifier.
2. Random Forest – bagging ensemble of decision trees.
3. Gradient Boosting – sequential boosting ensemble that fits new trees to reduce previous errors.

## 4. Holdout test results

| Metric | Logistic Regression | Random Forest | Gradient Boosting |
|---|---:|---:|---:|
| Training accuracy | 92.77% | 99.02% | 95.62% |
| Test accuracy | 92.85% | **97.29%** | 95.70% |
| Phishing precision | 93.54% | **97.82%** | 96.14% |
| Phishing recall | 90.10% | **96.02%** | 94.08% |
| Phishing F1 | 91.79% | **96.91%** | 95.10% |
| Phishing ROC-AUC | 98.08% | **99.78%** | 99.34% |

## 5. Five-fold cross-validation on the training partition

| Model | Mean CV accuracy | Mean CV phishing F1 |
|---|---:|---:|
| Logistic Regression | 92.83% | 91.80% |
| Random Forest | **96.99%** | **96.58%** |
| Gradient Boosting | 95.04% | 94.34% |

## 6. Confusion matrices

Matrix row/column order is `[Phishing (-1), Legitimate (1)]`.

- Logistic Regression: `[[883, 97], [61, 1170]]`
- Random Forest: `[[941, 39], [21, 1210]]`
- Gradient Boosting: `[[922, 58], [37, 1194]]`

For Random Forest, the model correctly identified 941 of 980 phishing samples in the holdout set. It missed 39 phishing samples and incorrectly flagged 21 legitimate samples.

## 7. Model selection and conclusion

Random Forest is selected for the current deployment artifact because it achieved the highest mean 5-fold cross-validation phishing F1 (96.58%) and also the strongest holdout phishing F1 (96.91%), recall (96.02%), accuracy (97.29%) and ROC-AUC (99.78%) among these three baseline configurations.

Gradient Boosting performed well, but it did not outperform Random Forest in this experiment. Therefore, the current results do **not** support claiming that Gradient Boosting is the best model. It may still be worth tuning Gradient Boosting and evaluating it on a fresh untouched test set in a later experiment.

## 8. Deployment artifact

`model.pkl` contains the Random Forest model refitted on all 11,055 labelled rows after model selection. The reported test metrics above come from the independent 20% holdout evaluation before this final refit. The deployment model expects the 30 input features in the same order as the dataset columns after excluding `Id` and `Result`.

## 9. Limitations

- Results are specific to this dataset and random split; they do not guarantee the same performance on current real-world URLs.
- The dataset consists of pre-engineered categorical/ordinal values; live URL feature extraction can be noisy or unavailable.
- The 30-feature vector produced at inference must match the training feature order and value conventions exactly.
- Accuracy and ROC-AUC alone are not sufficient for production security decisions. Continue monitoring phishing recall, false negatives, drift and external validation performance.
