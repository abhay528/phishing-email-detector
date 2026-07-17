"""Feature extraction for phishing email detection.

We combine two complementary signals:

1. **Textual content** - captured downstream with a TF-IDF vectorizer in the
   training pipeline (see ``src/train.py``).
2. **Hand-engineered URL / heuristic features** - computed here. These encode
   domain knowledge about what phishing emails tend to look like (IP-address
   URLs, look-alike domains, urgency keywords, credential requests, etc.).

The key public function is :func:`numeric_feature_matrix`, which turns an
iterable of raw email strings into a 2-D NumPy array. It is referenced by name
inside the scikit-learn pipeline, so it must remain importable for the saved
model to load correctly.
"""
from __future__ import annotations

import re
from typing import Iterable, List

import numpy as np

# ---------------------------------------------------------------------------
# Regexes and keyword lists
# ---------------------------------------------------------------------------

URL_REGEX = re.compile(r"https?://[^\s<>\"')]+", re.IGNORECASE)
IP_IN_URL_REGEX = re.compile(r"https?://(?:\d{1,3}\.){3}\d{1,3}", re.IGNORECASE)
EMAIL_REGEX = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

# Words/phrases commonly used to create urgency or request credentials.
SUSPICIOUS_KEYWORDS = [
    "verify", "suspend", "urgent", "immediately", "confirm", "account",
    "password", "login", "click", "update", "security", "alert", "limited",
    "expire", "unauthorized", "locked", "billing", "invoice", "winner",
    "congratulations", "prize", "gift", "free", "claim", "bank", "ssn",
    "credential", "restore", "validate", "final notice", "act now",
]

# TLDs disproportionately associated with abuse / free registration.
SUSPICIOUS_TLDS = (".xyz", ".top", ".info", ".ru", ".tk", ".ml", ".ga", ".cf", ".gq")

URL_SHORTENERS = ("bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd")

# Order matters: this list defines the columns of the feature matrix.
FEATURE_NAMES: List[str] = [
    "num_urls",
    "has_ip_url",
    "max_url_length",
    "num_dots_in_url",
    "has_https",
    "has_http_only",
    "num_suspicious_tld",
    "has_url_shortener",
    "num_at_symbols",
    "num_digits_in_url",
    "suspicious_keyword_count",
    "num_exclamation",
    "num_uppercase_words",
    "text_length",
    "num_email_addresses",
]


def extract_urls(text: str) -> List[str]:
    """Return all http(s) URLs found in ``text``."""
    return URL_REGEX.findall(text or "")


def _has_suspicious_tld(url: str) -> bool:
    """True if the URL's host ends with a suspicious TLD.

    We inspect only the host portion so that path/query segments (e.g. a
    ``.info`` inside a path) do not trigger false positives.
    """
    host = re.sub(r"^https?://", "", url.lower()).split("/")[0]
    host = host.split("?")[0].split(":")[0]
    return any(host.endswith(t) for t in SUSPICIOUS_TLDS)


def _features_for_text(text: str) -> List[float]:
    text = text or ""
    lower = text.lower()
    urls = extract_urls(text)

    url_lengths = [len(u) for u in urls]
    dots_in_urls = [u.count(".") for u in urls]
    digits_in_urls = sum(ch.isdigit() for u in urls for ch in u)

    has_ip_url = 1.0 if IP_IN_URL_REGEX.search(text) else 0.0
    has_https = 1.0 if any(u.lower().startswith("https://") for u in urls) else 0.0
    has_http_only = 1.0 if any(
        u.lower().startswith("http://") for u in urls
    ) else 0.0
    num_suspicious_tld = float(sum(_has_suspicious_tld(u) for u in urls))
    has_shortener = 1.0 if any(
        s in u.lower() for u in urls for s in URL_SHORTENERS
    ) else 0.0

    keyword_count = float(sum(lower.count(k) for k in SUSPICIOUS_KEYWORDS))
    uppercase_words = float(
        sum(1 for w in text.split() if len(w) > 2 and w.isupper())
    )

    return [
        float(len(urls)),                       # num_urls
        has_ip_url,                             # has_ip_url
        float(max(url_lengths) if url_lengths else 0),   # max_url_length
        float(max(dots_in_urls) if dots_in_urls else 0),  # num_dots_in_url
        has_https,                              # has_https
        has_http_only,                          # has_http_only
        num_suspicious_tld,                     # num_suspicious_tld
        has_shortener,                          # has_url_shortener
        float(text.count("@")),                 # num_at_symbols
        float(digits_in_urls),                  # num_digits_in_url
        keyword_count,                          # suspicious_keyword_count
        float(text.count("!")),                 # num_exclamation
        uppercase_words,                        # num_uppercase_words
        float(len(text)),                       # text_length
        float(len(EMAIL_REGEX.findall(text))),  # num_email_addresses
    ]


def numeric_feature_matrix(texts: Iterable[str]) -> np.ndarray:
    """Convert an iterable of raw email strings into a numeric feature matrix.

    This function is used inside the scikit-learn pipeline via
    ``FunctionTransformer`` and must therefore accept an array-like of strings
    and return a 2-D array with one row per input.
    """
    if hasattr(texts, "tolist"):
        texts = texts.tolist()
    return np.array([_features_for_text(t) for t in texts], dtype=float)


if __name__ == "__main__":
    demo = [
        "Subject: Verify now\n\nYour account is suspended! Click http://192.168.0.1/login",
        "Subject: Lunch?\n\nAre you free tomorrow? See notes at https://github.com/x/y",
    ]
    matrix = numeric_feature_matrix(demo)
    for name, value in zip(FEATURE_NAMES, matrix[0]):
        print(f"{name:>26}: {value}")
