# 🛡️ Phishing Email Detection with Machine Learning

A machine learning system that classifies emails as **Phishing** or **Safe**
using scikit-learn. It combines **TF-IDF text features** with **hand-engineered
URL and heuristic features** (IP-address links, look-alike domains, urgency
keywords, suspicious TLDs, URL shorteners, and more) to detect phishing based
on both *what an email says* and *what links it contains*.

> Built as a portfolio / resume project demonstrating an end-to-end ML workflow:
> data preparation → feature engineering → model training & comparison →
> evaluation → inference.

---

## ✨ Features

- **Trains on phishing + legitimate emails** — ships with a self-contained
  synthetic dataset generator, and drops in any real CSV with the same columns.
- **Rich feature extraction** — TF-IDF (word + bigram) over the email body plus
  15 engineered URL/heuristic features.
- **Two models compared** — Logistic Regression vs. Random Forest; the
  best-performing model (by F1) is automatically saved.
- **Clear evaluation** — prints **accuracy, precision, recall, F1** and a full
  classification report, and saves a **confusion matrix** heatmap.
- **Ready-to-use inference** — classify any email from the command line.

---

## 📁 Project Structure

```
phishing-email-detector/
├── data/
│   └── emails.csv              # dataset (generated or your own)
├── src/
│   ├── generate_dataset.py     # create a synthetic email dataset
│   ├── feature_extraction.py   # URL + keyword feature engineering
│   ├── train.py                # train, evaluate, save model & plots
│   └── predict.py              # classify new emails
├── outputs/
│   ├── confusion_matrix.png    # generated after training
│   └── metrics.json            # generated after training
├── models/
│   └── phishing_model.joblib   # generated after training
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Quickstart

```bash
# 1. Install dependencies (ideally in a virtual environment)
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt

# 2. Generate the dataset (or supply your own data/emails.csv)
python -m src.generate_dataset --n 1200 --out data/emails.csv

# 3. Train and evaluate the model
python -m src.train --data data/emails.csv

# 4. Classify new emails
python -m src.predict --demo
python -m src.predict --text "Your account is suspended! Verify at http://1.2.3.4/login"
```

> Run all commands from the project root so that the `src` package imports
> resolve correctly.

---

## 🧠 How It Works

### 1. Feature engineering (`src/feature_extraction.py`)

Each email is turned into two complementary representations:

| Representation | Description |
| --- | --- |
| **TF-IDF** | Word + bigram term-frequency features over the email text. |
| **Engineered features** | 15 numeric signals derived from the raw email. |

The engineered features include:

- `num_urls`, `max_url_length`, `num_dots_in_url`, `num_digits_in_url`
- `has_ip_url` (links pointing to raw IP addresses)
- `has_https`, `has_http_only`
- `num_suspicious_tld` (e.g. `.xyz`, `.top`, `.tk`), `has_url_shortener`
- `suspicious_keyword_count` (verify, urgent, suspend, password, …)
- `num_exclamation`, `num_uppercase_words`, `num_at_symbols`
- `text_length`, `num_email_addresses`

### 2. Model pipeline (`src/train.py`)

A single scikit-learn `Pipeline` joins the two feature families with a
`FeatureUnion`, then feeds a classifier:

```
raw email text
      │
      ├── TfidfVectorizer ─────────────┐
      │                                ├── FeatureUnion ── Classifier ── Safe / Phishing
      └── FunctionTransformer + Scaler ┘
```

Because feature extraction lives *inside* the pipeline, inference is just
`model.predict([raw_text])` — no manual preprocessing required.

### 3. Evaluation

Training prints metrics for both models and a classification report for the
best one, and writes `outputs/confusion_matrix.png` and `outputs/metrics.json`.

---

## 🔄 Using Your Own Dataset

Replace `data/emails.csv` with any CSV containing these columns:

| Column | Description |
| --- | --- |
| `text` | The full email text (subject + body). |
| `label` | `phishing` or `safe`. |
| `sender` | *(optional)* sender address. |

Public datasets you can adapt: the **Enron** corpus (legitimate) combined with
**Nazario / PhishTank** phishing samples, or Kaggle phishing-email datasets.

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **scikit-learn** — pipeline, vectorization, models, metrics
- **pandas / NumPy** — data handling
- **matplotlib** — confusion matrix visualization
- **joblib** — model persistence

---

## 📈 Possible Extensions

- Add more classifiers (SVM, Gradient Boosting) and hyperparameter tuning.
- Parse real `.eml` files and extract header-based features (SPF/DKIM, reply-to
  mismatch).
- Serve the model behind a small Flask/FastAPI endpoint.
- Add cross-validation and a held-out validation split.

---

## 📄 License

Released under the [MIT License](LICENSE).
