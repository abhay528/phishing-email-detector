"""Generate a realistic synthetic dataset of phishing and legitimate emails.

This keeps the repository fully self-contained and runnable out of the box.
Each row contains an email `text` (subject + body), the `sender` address, and a
`label` ("phishing" or "safe"). You can replace data/emails.csv with a real
dataset that has the same columns and everything else will still work.

Usage:
    python -m src.generate_dataset --n 1200 --out data/emails.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import random

# ---------------------------------------------------------------------------
# Building blocks for synthetic emails
# ---------------------------------------------------------------------------

PHISHING_BRANDS = [
    "PayPal", "Apple", "Amazon", "Netflix", "Microsoft", "Bank of America",
    "Wells Fargo", "DHL", "FedEx", "Chase", "Google", "Facebook", "Instagram",
]

PHISHING_SUBJECTS = [
    "URGENT: Your {brand} account has been suspended",
    "Action Required: Verify your {brand} account now",
    "Your {brand} password will expire in 24 hours",
    "Security Alert: Unusual sign-in detected on your {brand} account",
    "You have (1) pending payment - confirm immediately",
    "Congratulations! You have won a $1000 {brand} gift card",
    "Final Notice: Update your billing information",
    "Your account has been locked due to suspicious activity",
    "Re: Invoice overdue - immediate payment required",
    "Confirm your identity to avoid account closure",
]

PHISHING_BODIES = [
    ("Dear Customer,\n\nWe detected unusual activity on your {brand} account. "
     "You must verify your information within 24 hours or your account will be "
     "permanently suspended. Click the link below to confirm your details:\n"
     "{url}\n\nFailure to act will result in account termination.\n\n{brand} Security Team"),
    ("Hello,\n\nYour payment could not be processed. To avoid a late fee, please "
     "update your billing information immediately by visiting: {url}\n\n"
     "This is your final notice. Act now!"),
    ("Congratulations!!!\n\nYou have been selected to receive a FREE reward. "
     "Claim your prize now before it expires: {url}\n\nHurry, offer valid for "
     "the next 2 hours only."),
    ("Dear user,\n\nWe were unable to verify your account information. Please "
     "re-enter your login and password here to restore access: {url}\n\n"
     "Your account will be disabled if you do not respond."),
    ("Security Notice:\n\nSomeone tried to sign in to your account from a new "
     "device. If this wasn't you, secure your account immediately: {url}\n\n"
     "Verify now to prevent unauthorized access."),
]

LEGIT_SUBJECTS = [
    "Meeting notes from today's standup",
    "Your order has shipped",
    "Weekly project update",
    "Lunch tomorrow?",
    "Reminder: Team retrospective on Friday",
    "Draft of the quarterly report attached",
    "Thanks for your help last week",
    "Question about the assignment",
    "Invitation: Design review",
    "Your monthly newsletter",
    "Follow up on our conversation",
    "Updated slides for the presentation",
]

LEGIT_BODIES = [
    ("Hi {name},\n\nThanks for joining the call earlier. I've summarized the "
     "action items and shared them in the doc. Let me know if I missed "
     "anything.\n\nBest,\nAlex"),
    ("Hello {name},\n\nJust a reminder that our team retrospective is scheduled "
     "for Friday at 3 PM. Please add your discussion points to the shared "
     "agenda beforehand.\n\nCheers,\nJordan"),
    ("Hi team,\n\nHere is this week's progress update. We completed the API "
     "integration and started testing. More details in the linked document: "
     "{url}\n\nThanks,\nSam"),
    ("Hey {name},\n\nAre you free for lunch tomorrow around noon? Thinking of "
     "trying the new place near the office. Let me know!\n\nTalk soon,\nRiley"),
    ("Hi {name},\n\nYour order #{num} has shipped and should arrive within 3-5 "
     "business days. You can track it here: {url}\n\nThank you for your "
     "purchase.\n\nCustomer Support"),
    ("Hello {name},\n\nPlease find attached the draft of the quarterly report. "
     "I'd appreciate your feedback before we finalize it on Monday.\n\n"
     "Regards,\nTaylor"),
]

SAFE_DOMAINS = [
    "github.com", "docs.google.com", "company.com", "university.edu",
    "dropbox.com", "notion.so", "linkedin.com", "wikipedia.org",
]

SAFE_SENDER_DOMAINS = [
    "gmail.com", "outlook.com", "company.com", "university.edu", "icloud.com",
]

NAMES = ["Chris", "Pat", "Morgan", "Casey", "Jamie", "Dana", "Lee", "Robin"]

SUSPICIOUS_TLDS = ["xyz", "top", "info", "ru", "tk", "ml", "ga"]
SHORTENER_SLUGS = ["3xK9", "2aZ", "winPrize", "urgent", "claimNow"]


def _random_safe_url() -> str:
    domain = random.choice(SAFE_DOMAINS)
    path = random.choice(["", "/document/123", "/track/order", "/report", "/pull/42"])
    return "https://" + domain + path


def _random_phishing_url() -> str:
    # Hallmarks of phishing URLs: IP hosts, look-alike domains, odd TLDs,
    # excessive subdomains, http (not https), shorteners, and query strings.
    # Built with plain concatenation to keep URLs clean and readable.
    style = random.randint(0, 4)
    brand = random.choice(PHISHING_BRANDS).lower().replace(" ", "")
    tld = random.choice(SUSPICIOUS_TLDS)
    if style == 0:
        ip = ".".join(str(random.randint(1, 255)) for _ in range(4))
        return "http://" + ip + "/login/verify.php?id=" + str(random.randint(1000, 9999))
    if style == 1:
        return "http://" + brand + "-secure-login." + tld + "/verify"
    if style == 2:
        return ("http://account-" + brand + "." + str(random.randint(10, 99))
                + ".verify-now.com/update?token=abc" + str(random.randint(100, 999)))
    if style == 3:
        slug = random.choice(SHORTENER_SLUGS)
        return "http://bit.ly/" + slug + str(random.randint(10, 99))
    return "https://" + brand + ".com.secure-account-update." + tld + "/signin"


def _make_phishing_row() -> dict:
    brand = random.choice(PHISHING_BRANDS)
    subject = random.choice(PHISHING_SUBJECTS).format(brand=brand)
    body = random.choice(PHISHING_BODIES).format(brand=brand, url=_random_phishing_url())
    sender_brand = brand.lower().replace(" ", "")
    sender = (random.choice(["support", "security", "no-reply", "service"]) + "@"
              + sender_brand + "-" + random.choice(["secure", "verify", "alert"]) + "."
              + random.choice(["xyz", "top", "info", "com"]))
    text = "Subject: " + subject + "\n\n" + body
    return {"text": text, "sender": sender, "label": "phishing"}


def _make_safe_row() -> dict:
    subject = random.choice(LEGIT_SUBJECTS)
    name = random.choice(NAMES)
    body = random.choice(LEGIT_BODIES).format(
        name=name, url=_random_safe_url(), num=random.randint(10000, 99999)
    )
    sender = (name.lower() + "." + random.choice(["smith", "lee", "garcia", "khan"])
              + "@" + random.choice(SAFE_SENDER_DOMAINS))
    text = "Subject: " + subject + "\n\n" + body
    return {"text": text, "sender": sender, "label": "safe"}


def generate(n: int, seed: int = 42) -> list:
    random.seed(seed)
    rows = []
    for _ in range(n // 2):
        rows.append(_make_phishing_row())
        rows.append(_make_safe_row())
    random.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic email dataset")
    parser.add_argument("--n", type=int, default=1200, help="Total number of emails")
    parser.add_argument("--out", default="data/emails.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = generate(args.n, args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "sender", "label"])
        writer.writeheader()
        writer.writerows(rows)

    n_phish = sum(1 for r in rows if r["label"] == "phishing")
    print("Wrote " + str(len(rows)) + " emails to " + args.out
          + " (" + str(n_phish) + " phishing, " + str(len(rows) - n_phish) + " safe)")


if __name__ == "__main__":
    main()
