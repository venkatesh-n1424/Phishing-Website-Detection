import os
import pickle
import ipaddress
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for
from feature import FeatureExtraction

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-before-production")

# Load the selected, retrained Random Forest model once when the application starts.
with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)


def normalize_url(raw_url):
    value = (raw_url or "").strip()
    if not value:
        raise ValueError("Enter a URL to scan.")
    if not value.startswith(("http://", "https://")):
        value = "https://" + value
    parsed = urlparse(value)
    if parsed.username or parsed.password:
        raise ValueError("URLs containing credentials are not allowed.")

    if parsed.port not in (None, 80, 443):
        raise ValueError("Unsupported port.")
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("Enter a valid http:// or https:// URL.")
    # Do not allow IP-literal targets in this first version.
    try:
        ipaddress.ip_address(parsed.hostname or "")
        raise ValueError("Scanning direct IP-address URLs is not supported.")
    except ValueError as exc:
        if str(exc).startswith("Scanning"):
            raise
    return value


@app.route("/")
@app.route("/first")
def first():
    return render_template("first.html")


@app.route("/index")
def index():
    return render_template("index.html")


@app.route("/posts", methods=["GET", "POST"])
def posts():
    if request.method == "GET":
        return render_template("index.html")

    try:
        target_url = normalize_url(request.form.get("url"))
        extractor = FeatureExtraction(target_url)
        features = extractor.getFeaturesList()

        if len(features) != 30:
            raise RuntimeError(
                f"Feature extraction returned {len(features)} features; the model expects 30."
            )

        feature_vector = np.asarray(features, dtype=float).reshape(1, 30)
        prediction = int(model.predict(feature_vector)[0])

        probabilities = model.predict_proba(feature_vector)[0]
        classes = list(model.classes_)
        probability_by_class = {
            int(label): float(probability)
            for label, probability in zip(classes, probabilities)
        }

        # This project's original feature convention is 1 = legitimate, -1 = phishing.
        phishing_probability = probability_by_class.get(-1, 0.0)
        legitimate_probability = probability_by_class.get(1, 0.0)
        is_phishing = prediction == -1

        return render_template(
            "result.html",
            scanned_url=target_url,
            is_phishing=is_phishing,
            phishing_probability=round(phishing_probability * 100, 1),
            legitimate_probability=round(legitimate_probability * 100, 1),
        )
    except Exception as exc:
        app.logger.exception("URL scan failed")
        flash(
            "The scan could not be completed. Check the URL and try again. "
            "Some website or WHOIS lookups may be unavailable.",
            "error",
        )
        return redirect(url_for("index"))


@app.route("/performance")
def performance():
    return render_template("performance.html")


@app.route("/chart")
def chart():
    return render_template("chart.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    # The original project did not include authentication credentials or user storage.
    flash("This demo does not require an account. Use the URL Scanner instead.", "info")
    return redirect(url_for("index"))


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        uploaded = request.files.get("datasetfile")
        if not uploaded or not uploaded.filename.lower().endswith(".csv"):
            flash("Choose a CSV file to preview.", "error")
            return redirect(url_for("upload"))
        try:
            df = pd.read_csv(uploaded, encoding="utf-8", nrows=100)
            return render_template(
                "preview.html",
                columns=list(df.columns),
                rows=df.fillna("").astype(str).values.tolist(),
                row_count=len(df),
            )
        except Exception:
            app.logger.exception("CSV preview failed")
            flash("The CSV could not be read. Please check its format.", "error")
            return redirect(url_for("upload"))
    return render_template("upload.html")


@app.errorhandler(404)
def not_found(_error):
    return render_template("404.html"), 404


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=True)
