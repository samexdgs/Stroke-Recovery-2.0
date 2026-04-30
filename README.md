https://stroketracker2.streamlit.app/
# Stroke Recovery Monitor v2

Family-connected stroke recovery monitoring with machine learning and email alerts.

This is the second tool in a series built by Samuel Oluwakoya, a computer science graduate and foot drop patient building ML-based rehabilitation systems. Version 2 came out of a specific problem: many stroke survivors in Nigeria have children living abroad who cannot physically check on them every day. The monitor needed two roles, not one.

---
## The problem this solves

Stroke survivors often live alone or with elderly partners while their adult children are in a different country. A daily check-in system is only useful if the results reach the people who would act on them. This tool closes that loop. The patient logs their data in two minutes. Every family member who registered gets an email showing the recovery status, blood pressure readings, exercises completed, and any alerts the system flagged — all automatically.

---
## Two roles, one system

**Patient role** handles daily data entry and sees their own recommendations and progress charts.

**Family role** is read-only. Family members see everything the patient logged, receive email alerts when something needs attention, and can download the full recovery history as a CSV. They cannot modify patient data.

---

## What the ML model does

Three classifiers are trained on 2,000 simulated stroke survivor records. The best-performing model by ROC-AUC is selected automatically and applied to each daily entry.

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | ~90% | ~0.97 |
| Random Forest | ~91% | ~0.98 |
| LightGBM | ~92% | ~0.99 |

18 clinical features are used: age, days since stroke, affected side, stroke type, pain, fatigue, spasticity, balance, mobility, exercises completed, exercise duration, sleep quality, mood, AFO use, cane use, walker use, systolic BP, and diastolic BP.

The recommendation engine sits on top of the ML output and adds specific daily guidance. High pain triggers modified exercise protocols. Poor sleep triggers spasticity-timing advice. Low mood triggers the post-stroke depression clinical pathway.

---
## Email alerts

Alerts are sent automatically when blood pressure exceeds a safe threshold, when recovery status is classified as Needs Attention, and when a daily check-in is completed. The full summary is included in each alert so family members do not need to open the app to understand what happened.
---
## Tech stack

- Python 3.10
- Streamlit (web interface and multi-role session management)
- scikit-learn (Logistic Regression, Random Forest)
- LightGBM
- smtplib (email alerts via Gmail SMTP)
- Plotly (charts)
- pandas, numpy
- SHA-256 password hashing (no plain-text credentials stored)


## Project structure

```
stroke-recovery-monitor/
├── app.py          Main Streamlit application
├── database.py     JSON-based patient data storage
├── ml_engine.py    ML models and recommendations engine
├── alerts.py       Email alert system
├── requirements.txt
└── data/           Auto-created on first run
```
## Where this fits in the wider project

1. Foot Drop Management App — live at fdmapp.streamlit.app
2. Stroke Recovery Progress Tracker — daily monitoring, single user
3. Stroke Recovery Monitor v2 (this tool) — adds family dashboard and email alerts
4. AFO Clinical Management Platform — physiotherapist and patient dual dashboard with ML prescription
5. NeuroKinetics — camera-based upper limb motor tracking, no wearables needed

---
## Academic reference

Samuel Oluwakoya (2026). Stroke Recovery Monitor v2: A Multi-User Family-Connected Machine Learning Platform for Community Stroke Rehabilitation. GitHub. (https://github.com/samexdgs/Stroke-Recovery-2.0

- Email: [soluwakoyat@gmail.com](mailto:soluwakoyat@gmail.com),
- ORCID: [0009-0008-2126-0254](https://orcid.org/0009-0008-2126-0254)
- GitHub: [github.com/samexdgs](https://github.com/samexdgs)
- LinkedIn: [linkedin.com/in/samueloluwakoya](https://linkedin.com/in/samueloluwakoya)
- Portfolio: [samueloluwakoya.netlify.app](https://samueloluwakoya.netlify.app)

---
## Disclaimer

Research tool only. Not validated as a medical device. Does not constitute clinical advice. Always follow the guidance of your physiotherapist or rehabilitation physician.
---
Samuel Oluwakoya — computer science graduate, foot drop patient, AI health researcher.
