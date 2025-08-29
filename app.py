import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import requests
from bs4 import BeautifulSoup
import whois
from datetime import datetime
import re
from flask import Flask, request, render_template, jsonify
import warnings
warnings.filterwarnings('ignore')

# ================================
# FEATURE COLUMNS
# ================================
FEATURE_COLUMNS = [
    'url_length', 'num_dots', 'num_hyphens', 'num_underscores',
    'num_slashes', 'num_digits', 'num_special_chars',
    'has_ip', 'has_suspicious_tld', 'has_https',
    'has_www', 'suspicious_keywords'
]

# ================================
# STEP 1: Create Sample Dataset
# ================================
def create_sample_dataset():
    legitimate_urls = [
        'google.com', 'facebook.com', 'amazon.com', 'microsoft.com', 'apple.com',
        'wikipedia.org', 'youtube.com', 'twitter.com', 'linkedin.com', 'instagram.com',
        'github.com', 'stackoverflow.com', 'reddit.com', 'netflix.com', 'spotify.com',
        'adobe.com', 'paypal.com', 'ebay.com', 'cnn.com', 'bbc.com'
    ]
    suspicious_urls = [
        'paypaI-security-update.com', 'amazon-security.tk', 'facebook-login.ml',
        'google-verify.cf', 'microsoft-update.ga', 'paypal-dispute.tk',
        'bank-security-alert.com', 'apple-id-verification.ml', 'netflix-billing.cf',
        'secure-paypal-update.ga', 'amazon-refund.tk', 'facebook-security.ml',
        'google-account-suspended.cf', 'microsoft-security-alert.ga', 'apple-support-team.tk',
        'paypal-account-limited.ml', 'amazon-customer-service.cf', 'bank-account-verification.ga',
        'secure-login-required.tk', 'account-suspended-verify.ml'
    ]
    data = []
    for url in legitimate_urls:
        data.append({'url': url, 'label': 0})
    for url in suspicious_urls:
        data.append({'url': url, 'label': 1})
    return pd.DataFrame(data)

# ================================
# STEP 2: Feature Extraction
# ================================
def extract_url_features(url):
    features = {}
    features['url_length'] = len(url)
    features['num_dots'] = url.count('.')
    features['num_hyphens'] = url.count('-')
    features['num_underscores'] = url.count('_')
    features['num_slashes'] = url.count('/')
    features['num_digits'] = sum(c.isdigit() for c in url)
    features['num_special_chars'] = len(re.findall(r'[^a-zA-Z0-9./:-]', url))
    features['has_ip'] = 1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0
    features['has_suspicious_tld'] = 1 if any(tld in url.lower() for tld in ['.tk', '.ml', '.cf', '.ga', '.click']) else 0
    features['has_https'] = 1 if url.startswith('https') else 0
    features['has_www'] = 1 if 'www.' in url else 0
    features['suspicious_keywords'] = len(re.findall(r'(security|update|verify|suspend|alert|billing|refund)', url.lower()))
    return [features[col] for col in FEATURE_COLUMNS]

# ================================
# STEP 3: Train / Load ML Model
# ================================
MODEL_FILE = "model.pkl"

def train_model():
    df = create_sample_dataset()
    X = [extract_url_features(url) for url in df['url']]
    y = df['label'].values
    X = np.array(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"✅ Model trained with {acc*100:.1f}% accuracy")
    joblib.dump(model, MODEL_FILE)
    return model

if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
    print("🔄 Loaded existing model.")
else:
    model = train_model()

# ================================
# STEP 4: Content + WHOIS Analysis
# ================================
def analyze_website_content(url):
    try:
        if not url.startswith('http'):
            url = 'http://' + url
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, timeout=10, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text().lower()
        analysis = {
            'word_count': len(text.split()),
            'title': soup.title.string if soup.title else "",
            'suspicious_word_count': sum(1 for word in ['urgent','verify now','suspended','click here','limited time','act now'] if word in text),
            'form_count': len(soup.find_all('form')),
            'password_fields': len(soup.find_all('input', {'type': 'password'})),
            'external_links': sum(1 for link in soup.find_all('a', href=True) if link['href'].startswith('http') and url.split('/')[2] not in link['href'])
        }
        return analysis
    except Exception as e:
        return {'error': str(e)}

def analyze_whois_info(domain):
    try:
        domain = domain.replace('http://','').replace('https://','').split('/')[0]
        w = whois.whois(domain)
        analysis = {}
        if w.creation_date:
            creation_date = w.creation_date[0] if isinstance(w.creation_date, list) else w.creation_date
            if creation_date:
                age_days = (datetime.now() - creation_date).days
                analysis['domain_age_days'] = age_days
                analysis['is_very_new'] = age_days < 30
                analysis['is_new'] = age_days < 365
        analysis['registrar'] = str(w.registrar) if w.registrar else "Unknown"
        analysis['country'] = str(w.country) if w.country else "Unknown"
        return analysis
    except Exception as e:
        return {'error': str(e)}

# ================================
# STEP 5: Main Detection
# ================================
def detect_fake_website(url):
    results = {
        'url': url,
        'ml_score': 0, 'content_score': 0, 'whois_score': 0,
        'final_score': 0, 'risk_level': 'Unknown', 'reasons': []
    }
    try:
        features = extract_url_features(url)
        ml_prediction = model.predict_proba([features])[0]
        results['ml_score'] = ml_prediction[1]*100
        if results['ml_score'] > 70: results['reasons'].append("ML: Highly suspicious URL structure")
        elif results['ml_score'] > 40: results['reasons'].append("ML: Some suspicious patterns detected")
    except Exception as e:
        results['reasons'].append(f"ML error: {e}")

    content = analyze_website_content(url)
    if 'error' not in content:
        if content.get('suspicious_word_count',0) > 2: results['reasons'].append("Content: Suspicious language found")
        if content.get('password_fields',0) > 1: results['reasons'].append("Content: Multiple password fields detected")
        results['content_score'] = min((content.get('suspicious_word_count',0)+content.get('password_fields',0))*20,100)
    else:
        results['content_score'] = 30
        results['reasons'].append("Content could not be analyzed")

    whois_info = analyze_whois_info(url)
    if 'error' not in whois_info:
        if whois_info.get('is_very_new'): results['reasons'].append("WHOIS: Very new domain")
        elif whois_info.get('is_new'): results['reasons'].append("WHOIS: Relatively new domain")
        results['whois_score'] = 20 if whois_info.get('is_new') else 0
    else:
        results['whois_score'] = 20
        results['reasons'].append("WHOIS unavailable")

    results['final_score'] = results['ml_score']*0.5 + results['content_score']*0.3 + results['whois_score']*0.2
    if results['final_score'] >= 70: results['risk_level'] = "HIGH RISK"
    elif results['final_score'] >= 40: results['risk_level'] = "MEDIUM RISK"
    else: results['risk_level'] = "LOW RISK"
    return results

# ================================
# STEP 6: Flask App
# ================================
app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def home():
    results = None
    if request.method == "POST":
        url = request.form.get("url")
        if url:
            results = detect_fake_website(url)
    return render_template("index.html", results=results)

@app.route("/api/detect", methods=["POST"])
def api_detect():
    data = request.json
    url = data.get("url")
    if not url:
        return jsonify({"error":"URL missing"}),400
    results = detect_fake_website(url)
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
