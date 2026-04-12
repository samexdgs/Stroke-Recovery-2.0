"""
ml_engine.py — Stroke Recovery Monitor v2.0
============================================
Trains three ML classifiers on a simulated stroke recovery dataset.
Exercises now use real uploaded Lottie JSON files from assets/.

Author: Samuel Oluwakoya
"""

import os
import base64
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb
import streamlit as st


# ── Lottie asset loader ──────────────────────────────────────────────────────

_ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def _b64(filename: str) -> str:
    path = os.path.join(_ASSET_DIR, filename)
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


_LOTTIE = {
    "ankle_pumps":     _b64("ankle_pumps.json"),
    "knee_slides":     _b64("knee_slides.json"),
    "weight_shifts":   _b64("weight_shifts.json"),
    "seated_marching": _b64("seated_marching.json"),
    "ankle_circles":   _b64("ankle_circles.json"),
    "wall_pushups":    _b64("wall_pushups.json"),
    "calf_raises":     _b64("calf_raises.json"),
    "deep_breathing":  _b64("deep_breathing.json"),
}


def get_exercise_lottie(key: str) -> str:
    return _LOTTIE.get(key, "")


# ── Dataset ──────────────────────────────────────────────────────────────────

def _generate_dataset(n=2000, seed=42):
    np.random.seed(seed)
    age           = np.random.randint(35, 90, n)
    days_post     = np.random.randint(7, 730, n)
    affected_side = np.random.choice([0, 1], n)
    stroke_type   = np.random.choice([0, 1], n)
    pain          = np.random.randint(1, 11, n)
    fatigue       = np.random.randint(1, 11, n)
    spasticity    = np.random.randint(1, 11, n)
    balance       = np.random.randint(1, 11, n)
    mobility      = np.random.randint(1, 11, n)
    exercises_done= np.random.randint(0, 11, n)
    exercise_min  = np.random.randint(0, 91, n)
    sleep_quality = np.random.randint(1, 11, n)
    mood          = np.random.randint(1, 11, n)
    uses_afo      = np.random.choice([0, 1], n)
    uses_cane     = np.random.choice([0, 1], n)
    uses_walker   = np.random.choice([0, 1], n)
    bp_systolic   = np.random.randint(100, 200, n)
    bp_diastolic  = np.random.randint(60, 120, n)

    score = (
        (mobility * 1.5) + (balance * 1.3) + (exercises_done * 1.2)
        + (sleep_quality * 0.8) + (mood * 0.6)
        - (pain * 1.1) - (fatigue * 0.9) - (spasticity * 0.7)
        - (np.clip(bp_systolic - 120, 0, 80) * 0.1)
        + (days_post * 0.01) - (age * 0.04)
        + np.random.normal(0, 3, n)
    )
    labels = pd.cut(score, bins=[-np.inf, 11, 21, np.inf], labels=[0, 1, 2]).astype(int)

    return pd.DataFrame({
        "age": age, "days_post": days_post,
        "affected_side": affected_side, "stroke_type": stroke_type,
        "pain": pain, "fatigue": fatigue, "spasticity": spasticity,
        "balance": balance, "mobility": mobility,
        "exercises_done": exercises_done, "exercise_min": exercise_min,
        "sleep_quality": sleep_quality, "mood": mood,
        "uses_afo": uses_afo, "uses_cane": uses_cane, "uses_walker": uses_walker,
        "bp_systolic": bp_systolic, "bp_diastolic": bp_diastolic,
        "recovery_status": labels
    })


FEATURES = [
    "age", "days_post", "affected_side", "stroke_type",
    "pain", "fatigue", "spasticity", "balance", "mobility",
    "exercises_done", "exercise_min", "sleep_quality", "mood",
    "uses_afo", "uses_cane", "uses_walker", "bp_systolic", "bp_diastolic"
]


# ── Model training ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Initialising recovery models...")
def load_models():
    df = _generate_dataset(2000)
    X, y = df[FEATURES], df["recovery_status"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(X_train)
    Xte_sc = scaler.transform(X_test)

    lr   = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(Xtr_sc, y_train)

    rf   = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    lgbm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                               random_state=42, verbose=-1)
    lgbm.fit(X_train, y_train)

    results = {
        "Logistic Regression": {
            "model": lr, "scaler": scaler, "uses_scaler": True,
            "accuracy": accuracy_score(y_test, lr.predict(Xte_sc)),
            "roc_auc":  roc_auc_score(y_test, lr.predict_proba(Xte_sc), multi_class="ovr"),
        },
        "Random Forest": {
            "model": rf, "scaler": None, "uses_scaler": False,
            "accuracy": accuracy_score(y_test, rf.predict(X_test)),
            "roc_auc":  roc_auc_score(y_test, rf.predict_proba(X_test), multi_class="ovr"),
        },
        "LightGBM": {
            "model": lgbm, "scaler": None, "uses_scaler": False,
            "accuracy": accuracy_score(y_test, lgbm.predict(X_test)),
            "roc_auc":  roc_auc_score(y_test, lgbm.predict_proba(X_test), multi_class="ovr"),
        },
    }
    best = max(results, key=lambda k: results[k]["roc_auc"])
    return results, best, FEATURES


def predict(model_results, best_name, input_dict):
    r = model_results[best_name]
    X = pd.DataFrame([input_dict])[FEATURES]
    if r["uses_scaler"]:
        X = r["scaler"].transform(X)
    return int(r["model"].predict(X)[0]), r["model"].predict_proba(X)[0].tolist()


# ── Constants ─────────────────────────────────────────────────────────────────

STATUS_LABELS = {0: "Needs Attention 🔴", 1: "Plateauing 🟡", 2: "On Track 🟢"}
STATUS_COLORS = {0: "#dc2626", 1: "#d97706", 2: "#059669"}
STATUS_BG     = {0: "#fee2e2", 1: "#fef3c7", 2: "#d1fae5"}


def check_bp_alert(systolic: int, diastolic: int):
    if systolic >= 180 or diastolic >= 120:
        return True, "critical", f"CRITICAL: BP {systolic}/{diastolic} mmHg — seek emergency care immediately"
    if systolic >= 140 or diastolic >= 90:
        return True, "high", f"HIGH: BP {systolic}/{diastolic} mmHg — above safe range, contact doctor today"
    if systolic < 90 or diastolic < 60:
        return True, "low", f"LOW: BP {systolic}/{diastolic} mmHg — below normal, rest and hydrate"
    return False, "normal", f"Normal: BP {systolic}/{diastolic} mmHg"


def get_recommendations(status: int, data: dict) -> list:
    recs = []
    if status == 0:
        recs.append({"icon": "🔴", "priority": "urgent",
            "title": "Contact your healthcare provider this week",
            "body": "Today's data suggests your recovery needs clinical review. "
                    "Please call your physiotherapist or rehabilitation doctor. "
                    "Show them the daily report from this app."})
    elif status == 1:
        recs.append({"icon": "🟡", "priority": "moderate",
            "title": "Plateau detected — time to change the stimulus",
            "body": "Your brain needs new challenges to continue rewiring. "
                    "Try a new exercise type or environment this week. "
                    "Ask your physiotherapist about electrical stimulation therapy."})
    else:
        recs.append({"icon": "🟢", "priority": "good",
            "title": "On track — consistency is everything",
            "body": "Every day of exercises, even a short session, builds the neural pathways "
                    "that become permanent. Keep it up."})

    is_alert, severity, bp_msg = check_bp_alert(
        data.get("bp_systolic", 120), data.get("bp_diastolic", 80))
    if is_alert:
        recs.append({"icon": "💉",
            "priority": "urgent" if severity == "critical" else "moderate",
            "title": f"Blood pressure alert — {severity.upper()}",
            "body": bp_msg + (". Rest immediately and call emergency services."
                               if severity == "critical"
                               else ". Rest, avoid exertion, take your medication and contact your doctor today.")})

    if data.get("exercises_done", 5) < 4:
        recs.append({"icon": "🏃", "priority": "moderate",
            "title": "Exercise completion is low today",
            "body": f"You completed {data.get('exercises_done',0)}/10 exercises. "
                    "Start with just 3: seated ankle pumps, knee slides, and standing weight shifts."})
    elif data.get("exercises_done", 5) >= 8:
        recs.append({"icon": "🏆", "priority": "good",
            "title": "Excellent exercise completion — consider progressing",
            "body": "Hitting 80%+ consistently. Try adding light resistance this week."})

    if data.get("pain", 3) >= 7:
        recs.append({"icon": "🌡️", "priority": "moderate",
            "title": "High pain — modify today's exercises",
            "body": "Pain at 7+ means switch to gentle range-of-motion only: ankle circles, toe curls, deep breathing. "
                    "If pain stays above 7 for 3+ days, your doctor must review."})
    if data.get("fatigue", 4) >= 7:
        recs.append({"icon": "😴", "priority": "moderate",
            "title": "High fatigue — use the pacing strategy",
            "body": "30-30-30 rule: 30 min light activity, 30 min rest, repeat. "
                    "Do NOT push through fatigue above 7."})
    if data.get("sleep_quality", 6) <= 4:
        recs.append({"icon": "🌙", "priority": "moderate",
            "title": "Poor sleep is slowing neurological recovery",
            "body": "Keep consistent sleep/wake times. No screens 1 hour before bed. "
                    "If spasticity wakes you at night, ask your doctor about medication timing."})
    if data.get("mood", 7) <= 4:
        recs.append({"icon": "🧠", "priority": "moderate",
            "title": "Low mood — this is a clinical symptom, not weakness",
            "body": "Post-stroke depression affects 30% of survivors. "
                    "Please mention your mood score to your medical team."})
    if data.get("spasticity", 3) >= 6:
        recs.append({"icon": "💪", "priority": "moderate",
            "title": "Spasticity is high — stretch before every exercise",
            "body": "Stretch the affected limb for 10–15 minutes before exercise. "
                    "A warm shower before stretching reduces muscle tone significantly."})
    return recs[:7]


# ── Exercise library — matched to uploaded Lottie JSONs ──────────────────────

EXERCISES = [
    {
        "name":         "Seated Ankle Pumps",
        "reps":         "10 reps × 3 sets",
        "duration":     "5 minutes",
        "difficulty":   "Beginner",
        "target":       "Ankle dorsiflexion, calf circulation",
        "instructions": (
            "Sit in a chair with feet flat on the floor. Slowly lift your toes toward your shin, "
            "hold 2 seconds, then point toes down. Perform slowly and with full control. "
            "Rest 30 seconds between sets. This trains the exact movement an AFO assists passively."
        ),
        "icon":       "🦶",
        "lottie_key": "ankle_pumps",
    },
    {
        "name":         "Knee Slides",
        "reps":         "10 reps each leg",
        "duration":     "8 minutes",
        "difficulty":   "Beginner",
        "target":       "Hip and knee flexion, hamstring length",
        "instructions": (
            "Lie on your back on a firm surface. Slowly slide one heel toward your bottom, "
            "bending the knee. Hold for 3 seconds, then slide back. "
            "Repeat on both sides. Keep movements slow — quality over speed."
        ),
        "icon":       "🦵",
        "lottie_key": "knee_slides",
    },
    {
        "name":         "Standing Weight Shifts",
        "reps":         "10 reps × 2 sets",
        "duration":     "6 minutes",
        "difficulty":   "Beginner",
        "target":       "Balance, weight-bearing on affected leg",
        "instructions": (
            "Stand holding a stable surface. Slowly shift your weight onto your affected leg, "
            "hold 3 seconds, then shift back to centre. Keep both feet flat throughout. "
            "Loosen your grip as you improve."
        ),
        "icon":       "⚖️",
        "lottie_key": "weight_shifts",
    },
    {
        "name":         "Seated Marching",
        "reps":         "20 reps alternating",
        "duration":     "5 minutes",
        "difficulty":   "Beginner",
        "target":       "Hip flexion, coordination, core activation",
        "instructions": (
            "Sit upright in a sturdy chair. Slowly lift one knee up, then lower and lift the other "
            "— like marching in place. Keep your back straight and breathe steadily. "
            "Hands rest lightly on thighs."
        ),
        "icon":       "🚶",
        "lottie_key": "seated_marching",
    },
    {
        "name":         "Ankle Circles",
        "reps":         "10 circles each direction, each foot",
        "duration":     "4 minutes",
        "difficulty":   "Beginner",
        "target":       "Ankle mobility, morning stiffness reduction",
        "instructions": (
            "Sit with feet lifted slightly off the floor. Rotate each ankle in large smooth circles "
            "— 10 clockwise, then 10 anti-clockwise. Complete one foot before the other. "
            "Excellent for reducing morning ankle stiffness."
        ),
        "icon":       "🔄",
        "lottie_key": "ankle_circles",
    },
    {
        "name":         "Wall Push-Ups",
        "reps":         "10 reps × 3 sets",
        "duration":     "7 minutes",
        "difficulty":   "Intermediate",
        "target":       "Upper body strength, shoulder stability",
        "instructions": (
            "Stand arm's length from a wall, hands flat at shoulder height. "
            "Slowly bend elbows toward wall, hold 2 seconds, push back. "
            "Focus on both arms doing equal work. Rest 45 seconds between sets."
        ),
        "icon":       "🤸",
        "lottie_key": "wall_pushups",
    },
    {
        "name":         "Calf Raises",
        "reps":         "10 reps × 2 sets",
        "duration":     "5 minutes",
        "difficulty":   "Intermediate",
        "target":       "Calf strength, ankle stability for walking",
        "instructions": (
            "Stand holding a chair back. Slowly rise up onto your toes, hold 2–3 seconds, "
            "then lower slowly over 4 counts. The slow lowering phase builds the most strength. "
            "If the affected leg is too weak alone, lead with the stronger leg but try to lower "
            "on the affected leg."
        ),
        "icon":       "🏋️",
        "lottie_key": "calf_raises",
    },
    {
        "name":         "Deep Breathing and Relaxation",
        "reps":         "5 minutes",
        "duration":     "5 minutes",
        "difficulty":   "All levels",
        "target":       "Nervous system regulation, blood pressure",
        "instructions": (
            "Sit or lie comfortably. Breathe in through your nose for 4 counts, "
            "hold for 2 counts, breathe out through your mouth for 6 counts. "
            "This activates the parasympathetic nervous system, reduces spasticity, "
            "and lowers blood pressure. Do this before bed to improve sleep quality."
        ),
        "icon":       "🧘",
        "lottie_key": "deep_breathing",
    },
]
