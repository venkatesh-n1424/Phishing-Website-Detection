# Phishing Website Detection

A Flask web application that uses a retrained Random Forest classifier to classify website URLs using 30 engineered URL and webpage features. Three algorithms (Logistic Regression, Random Forest and Gradient Boosting) are compared in the included experiment.

## Project structure

```text
Phishing-Website-Detection/
├── app.py
├── feature.py
├── model.pkl
├── requirements.txt
├── Procfile
├── runtime.txt
├── README.md
├── .gitignore
├── dataset/
│   └── upload.csv
├── docs/
│   ├── model_accuracy.txt
│   ├── project_report.pdf
│   └── project_presentation.ppt
├── static/
│   └── css/style.css
└── templates/
    ├── base.html
    ├── first.html
    ├── index.html
    ├── result.html
    ├── performance.html
    ├── chart.html
    ├── upload.html
    ├── preview.html
    └── 404.html
```

## Local setup (Windows PowerShell)

Use the Python version and scikit-learn version compatible with the supplied pickle model. The model was serialized in an older scikit-learn environment; if loading fails, do not blindly re-save it with a different version.

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000

If PowerShell blocks environment activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## GitHub

```bash
git init
git add .
git commit -m "Initial project setup"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/Phishing-Website-Detection.git
git push -u origin main
```

## Render deployment

- Create a new Web Service at https://render.com/
- Connect this GitHub repository.
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app`
- Set `SECRET_KEY` as a random environment variable in Render.

## Important project limitations

- The supplied ZIP did not contain any HTML or CSS files. This package recreates the user interface.
- The supplied model file may require its original scikit-learn version. The `requirements.txt` is a starting environment and must be validated against `model.pkl`.
- The original feature extraction code contains legacy references and external services that may fail or time out. Test feature extraction locally before publishing the service.
- The CSV upload route only previews the first 100 rows; it does not train or update the model.
- Reported accuracy (98.9% training / 97.4% testing) is transcribed from the supplied project file and has not been independently reproduced here.
- A URL classifier is not a browser safety system. Its output is not a guarantee that a website is safe or malicious. Do not submit passwords, payment data, or sensitive URLs to this demo.
- Before public deployment, add strict outbound-request controls, DNS/IP checks, timeouts, rate limiting, and other protections against server-side request forgery (SSRF).

## Model training and comparison

The project includes `train_models.py`, which trains and compares:
- Logistic Regression (with StandardScaler)
- Random Forest (400 trees)
- Gradient Boosting (150 estimators)

Run from the project root:

```bash
pip install -r training-requirements.txt
python train_models.py
```

The script uses a stratified 80/20 split (`random_state=42`) and 5-fold stratified cross-validation on the training partition. It selects the deployment model using mean cross-validation F1 for the phishing class (`Result == -1`), then refits that selected model on all labelled data and saves it as `model.pkl`.

**Current experiment result:** Random Forest was selected. On the held-out test set it achieved 97.29% accuracy, 97.82% phishing precision, 96.02% phishing recall, 96.91% phishing F1 and 99.78% phishing ROC-AUC. See `results/EXPERIMENT_REPORT.md` for full methodology, all model metrics and confusion matrices.

The original supplied Gradient Boosting model is preserved separately at `models/original_supplied_model.pkl`; it is not used by the application.

Input: 30 engineered URL and webpage features  
Classes: `1` = legitimate, `-1` = phishing (as documented in the original feature engineering code)

**Compatibility:** The current `model.pkl` was trained with scikit-learn 1.8.0. Use the versions in `requirements.txt` when loading this pickle. Pickle files are not guaranteed to be compatible across scikit-learn versions.
