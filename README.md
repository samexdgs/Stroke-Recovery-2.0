# 🧠 Stroke Recovery Monitor v2.0

**Family-connected, ML-powered stroke recovery tracking**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🌍 The Problem This Solves

*Mama had a stroke. Her children live abroad. They've arranged the best doctor — but they can't be there every day to see how she is doing.*

This tool solves that gap. Mama logs her daily recovery data in 2 minutes. Every child gets an instant email alert showing her blood pressure, exercises completed, recovery status, and personalised recommendations — all powered by machine learning.

**Two roles. One system.**

| Role | What they do |
|---|---|
| **Patient (Mama)** | Logs daily symptoms, BP, exercises, mood — gets personalised recommendations |
| **Family / Carers** | See real-time dashboard, receive email alerts, track full recovery timeline |

---

## ✨ Features

### For the Patient
- Daily check-in form (2 minutes, mobile-friendly)
- ML-powered recovery status: 🟢 On Track · 🟡 Plateauing · 🔴 Needs Attention
- Blood pressure logging with automatic alert detection
- Personalised daily recommendations (exercises, pain, fatigue, sleep, mood)
- 8-exercise guided rehabilitation library with full instructions
- Personal progress charts and data download

### For the Family
- Real-time monitoring dashboard (read-only)
- **Email alerts** when:
  - Blood pressure exceeds safe range
  - Recovery status is "Needs Attention"
  - Daily check-in is completed (with full summary)
- Full alert history with timestamps
- Downloadable recovery log (CSV)
- 30-day blood pressure trend chart
- Recovery status timeline

---

## 🤖 ML Architecture

Three classifiers trained in parallel on 2,000 simulated stroke survivor records:

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | ~90% | ~0.97 |
| Random Forest | ~91% | ~0.98 |
| **LightGBM** *(auto-selected)* | **~92%** | **~0.99** |

**18 clinical features:** Age · Days since stroke · Affected side · Stroke type · Pain · Fatigue · Spasticity · Balance · Mobility · Exercises done · Exercise duration · Sleep quality · Mood · AFO use · Cane use · Walker use · BP Systolic · BP Diastolic

---

## 🚀 Quick Start

### 1. Clone and install
```bash
git clone https://github.com/samueloluwakoya/stroke-recovery-monitor.git
cd stroke-recovery-monitor
pip install -r requirements.txt
```

### 2. Run locally
```bash
streamlit run app.py
```

### 3. Register a patient
Open the app → click "New patient? Register here" → fill in patient details, set a PIN and a family access code → share the family code with all family members.

---

## ☁️ Deploy to Streamlit Cloud

1. Push to a **public GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub → select repo → main file: `app.py`
4. Click **Deploy**

### Enable email alerts (optional)
In Streamlit Cloud → **Settings → Secrets**, add:
```toml
SENDER_EMAIL    = "your-gmail@gmail.com"
SENDER_PASSWORD = "your-16-char-app-password"
```
See `.streamlit/secrets.toml` for Gmail App Password setup instructions.

---

## 📁 File Structure

```
stroke-recovery-monitor/
├── app.py              # Main Streamlit application
├── database.py         # JSON-based patient data storage
├── ml_engine.py        # ML models + recommendations engine
├── alerts.py           # Email alert system
├── requirements.txt    # Python dependencies
├── .streamlit/
│   └── secrets.toml    # Email config (not committed to git)
└── data/               # Auto-created on first run
    ├── patients.json
    ├── logs.json
    └── alerts.json
```

---

## 🔒 Privacy & Security

- Patient PINs and family codes are stored as SHA-256 hashes — never as plain text
- Family view is **read-only** — family members cannot modify patient data
- All data stored locally in JSON files (Streamlit Cloud persistent storage)
- No data sent to third parties (email alerts only if configured)

---

## 📄 Academic Reference

> **Samuel Oluwakoya** (2026). *Stroke Recovery Monitor v2.0: A Multi-User, Family-Connected ML Platform for Community Stroke Rehabilitation*. GitHub. https://github.com/samueloluwakoya/stroke-recovery-monitor

Part of an ongoing research programme in ML-based neurological rehabilitation:

> Samuel Oluwakoya (2026). *ML-Based Drop-Foot Management System*. Live: https://fdmapp.streamlit.app

---

## 🗺️ Roadmap

- [ ] Real clinical dataset validation with ethics approval
- [ ] Wearable integration (step count, heart rate via Fitbit/Apple Health)
- [ ] WhatsApp alerts (via Twilio) in addition to email
- [ ] Missed-log reminder emails to family
- [ ] Multilingual interface (Yoruba, Hausa, Igbo)
- [ ] Android app wrapper

---

## ⚕️ Medical Disclaimer

This tool is for **informational and research purposes only**. It is not a validated medical device and does not constitute clinical advice. Always consult a qualified physiotherapist or rehabilitation physician. In an emergency, call emergency services immediately.

---

## 👨‍💻 About the Author

**Samuel Oluwakoya** — Computer science graduate, AI health researcher, and foot drop patient building machine learning tools for neurological rehabilitation.

*"I build these tools because I live with the condition they address."*

📧 Contact via GitHub Issues · 🔬 [ResearchGate](https://researchgate.net) · 💼 [LinkedIn](https://linkedin.com)
