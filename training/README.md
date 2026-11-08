# Model training

Run training from the project root:

```bash
python train_models.py
```

The script reads `dataset/upload.csv`, creates a stratified 80/20 split, evaluates Logistic Regression, Random Forest and Gradient Boosting, runs stratified 5-fold cross-validation on the training partition, and saves the comparison artifacts under `results/`.

Model selection uses mean cross-validation F1 for the phishing class (`Result == -1`) rather than choosing a model based on its test-set score. The selected model is refitted on all labelled rows and written to the root `model.pkl` for Flask inference.

Outputs include:
- `results/model_comparison.csv`
- `results/detailed_metrics.json`
- one confusion matrix per algorithm
- `results/roc_curve_comparison.png`
- feature-importance CSV files for the two tree ensembles
- `results/EXPERIMENT_REPORT.md`

The saved pickle must be loaded with a compatible scikit-learn version. The current artifact was trained using scikit-learn 1.8.0.
