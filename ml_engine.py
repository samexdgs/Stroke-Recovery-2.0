"""
ml_engine.py — Stroke Recovery Monitor v2.0
============================================
Trains three ML classifiers on a simulated stroke recovery dataset.
Returns predictions, confidence scores, and personalised recommendations.

Author: Samuel Oluwakoya
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb
import streamlit as st


# ─────────────────────────────────────────────
# DATASET GENERATION
# ─────────────────────────────────────────────

def _generate_dataset(n=2000, seed=42):
    np.random.seed(seed)

    age              = np.random.randint(35, 90, n)
    days_post        = np.random.randint(7, 730, n)
    affected_side    = np.random.choice([0, 1], n)
    stroke_type      = np.random.choice([0, 1], n)
    pain             = np.random.randint(1, 11, n)
    fatigue          = np.random.randint(1, 11, n)
    spasticity       = np.random.randint(1, 11, n)
    balance          = np.random.randint(1, 11, n)
    mobility         = np.random.randint(1, 11, n)
    exercises_done   = np.random.randint(0, 11, n)
    exercise_min     = np.random.randint(0, 91, n)
    sleep_quality    = np.random.randint(1, 11, n)
    mood             = np.random.randint(1, 11, n)
    uses_afo         = np.random.choice([0, 1], n)
    uses_cane        = np.random.choice([0, 1], n)
    uses_walker      = np.random.choice([0, 1], n)
    bp_systolic      = np.random.randint(100, 200, n)
    bp_diastolic     = np.random.randint(60, 120, n)

    score = (
        (mobility * 1.5) + (balance * 1.3) + (exercises_done * 1.2)
        + (sleep_quality * 0.8) + (mood * 0.6)
        - (pain * 1.1) - (fatigue * 0.9) - (spasticity * 0.7)
        - (np.clip(bp_systolic - 120, 0, 80) * 0.1)
        + (days_post * 0.01) - (age * 0.04)
        + np.random.normal(0, 3, n)
    )

    labels = pd.cut(score, bins=[-np.inf, 11, 21, np.inf],
                    labels=[0, 1, 2]).astype(int)

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
    "uses_afo", "uses_cane", "uses_walker",
    "bp_systolic", "bp_diastolic"
]


# ─────────────────────────────────────────────
# MODEL TRAINING (cached)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Initialising recovery models…")
def load_models():
    df = _generate_dataset(2000)
    X = df[FEATURES]
    y = df["recovery_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler    = StandardScaler()
    Xtr_sc    = scaler.fit_transform(X_train)
    Xte_sc    = scaler.transform(X_test)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(Xtr_sc, y_train)
    lr_auc = roc_auc_score(y_test, lr.predict_proba(Xte_sc), multi_class="ovr")
    lr_acc = accuracy_score(y_test, lr.predict(Xte_sc))

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test), multi_class="ovr")
    rf_acc = accuracy_score(y_test, rf.predict(X_test))

    # LightGBM
    lgbm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                               random_state=42, verbose=-1)
    lgbm.fit(X_train, y_train)
    lgbm_auc = roc_auc_score(y_test, lgbm.predict_proba(X_test), multi_class="ovr")
    lgbm_acc = accuracy_score(y_test, lgbm.predict(X_test))

    results = {
        "Logistic Regression": {"model": lr,   "scaler": scaler, "uses_scaler": True,
                                 "accuracy": lr_acc,   "roc_auc": lr_auc},
        "Random Forest":       {"model": rf,   "scaler": None,   "uses_scaler": False,
                                 "accuracy": rf_acc,   "roc_auc": rf_auc},
        "LightGBM":            {"model": lgbm, "scaler": None,   "uses_scaler": False,
                                 "accuracy": lgbm_acc, "roc_auc": lgbm_auc},
    }
    best = max(results, key=lambda k: results[k]["roc_auc"])
    return results, best, FEATURES


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────

def predict(model_results, best_name, input_dict):
    r = model_results[best_name]
    X = pd.DataFrame([input_dict])[FEATURES]
    if r["uses_scaler"]:
        X = r["scaler"].transform(X)
    pred  = int(r["model"].predict(X)[0])
    proba = r["model"].predict_proba(X)[0].tolist()
    return pred, proba


# ─────────────────────────────────────────────
# RECOMMENDATIONS ENGINE
# ─────────────────────────────────────────────

STATUS_LABELS  = {0: "Needs Attention 🔴", 1: "Plateauing 🟡", 2: "On Track 🟢"}
STATUS_COLORS  = {0: "#dc2626", 1: "#d97706", 2: "#059669"}
STATUS_BG      = {0: "#fee2e2", 1: "#fef3c7", 2: "#d1fae5"}


def check_bp_alert(systolic, diastolic):
    """Returns (is_alert, severity, message)"""
    if systolic >= 180 or diastolic >= 120:
        return True, "critical", f"CRITICAL: BP {systolic}/{diastolic} mmHg — seek emergency care immediately"
    if systolic >= 140 or diastolic >= 90:
        return True, "high", f"HIGH: BP {systolic}/{diastolic} mmHg — above safe range, contact doctor today"
    if systolic < 90 or diastolic < 60:
        return True, "low", f"LOW: BP {systolic}/{diastolic} mmHg — below normal, rest and hydrate"
    return False, "normal", f"Normal: BP {systolic}/{diastolic} mmHg"


def get_recommendations(status, data):
    recs = []

    # Status-level recommendation (always first)
    if status == 0:
        recs.append({
            "icon": "🔴", "priority": "urgent",
            "title": "Contact your healthcare provider this week",
            "body": "Today's data suggests your recovery needs clinical review. "
                    "Please call your physiotherapist or rehabilitation doctor. "
                    "Show them the daily report from this app."
        })
    elif status == 1:
        recs.append({
            "icon": "🟡", "priority": "moderate",
            "title": "Plateau detected — time to change the stimulus",
            "body": "Your brain needs new challenges to continue rewiring. "
                    "Try a new exercise type or environment this week. "
                    "Ask your physiotherapist about electrical stimulation therapy."
        })
    else:
        recs.append({
            "icon": "🟢", "priority": "good",
            "title": "You are on track — consistency is everything",
            "body": "Every day of exercises, even a short session, builds the neural pathways that become permanent. "
                    "Keep going — your family is cheering you on."
        })

    # Blood pressure
    is_alert, severity, bp_msg = check_bp_alert(data["bp_systolic"], data["bp_diastolic"])
    if is_alert:
        recs.append({
            "icon": "💉", "priority": "urgent" if severity == "critical" else "moderate",
            "title": f"Blood pressure alert — {severity.upper()}",
            "body": bp_msg + (". Rest immediately and call emergency services." if severity == "critical"
                              else ". Rest, avoid exertion, take your medication and contact your doctor today.")
        })

    # Exercise
    if data["exercises_done"] < 4:
        recs.append({
            "icon": "🏃", "priority": "moderate",
            "title": "Exercise completion is low today",
            "body": f"You completed {data['exercises_done']}/10 exercises. "
                    "Start with just 3: seated ankle pumps (10 reps × 3), knee slides, and standing weight-shifts. "
                    "Add one more each day until you reach your full plan."
        })
    elif data["exercises_done"] >= 8:
        recs.append({
            "icon": "🏆", "priority": "good",
            "title": "Excellent exercise completion — consider progressing",
            "body": "Hitting 80%+ of exercises consistently. "
                    "This week try adding light resistance: a resistance band for ankle dorsiflexion, "
                    "or increase standing balance time by 5 minutes."
        })

    # Pain
    if data["pain"] >= 7:
        recs.append({
            "icon": "🌡️", "priority": "moderate",
            "title": "High pain — modify today's exercises",
            "body": "Pain at 7+ means switch to gentle range-of-motion only: ankle circles, toe curls, deep breathing. "
                    "Apply heat for 15 minutes before exercise. "
                    "If pain stays above 7 for 3+ days, your doctor must review."
        })

    # Fatigue
    if data["fatigue"] >= 7:
        recs.append({
            "icon": "😴", "priority": "moderate",
            "title": "High fatigue — use the pacing strategy",
            "body": "Try the 30-30-30 rule: 30 min light activity, 30 min rest, repeat. "
                    "Do NOT push through fatigue above 7 — this causes setbacks."
        })

    # Sleep
    if data["sleep_quality"] <= 4:
        recs.append({
            "icon": "🌙", "priority": "moderate",
            "title": "Poor sleep is slowing neurological recovery",
            "body": "Sleep is when the brain consolidates motor learning. "
                    "Keep consistent sleep/wake times. No screens 1 hour before bed. "
                    "If spasticity wakes you at night, ask your doctor about medication timing."
        })

    # Mood
    if data["mood"] <= 4:
        recs.append({
            "icon": "🧠", "priority": "moderate",
            "title": "Low mood — this is a clinical symptom, not weakness",
            "body": "Post-stroke depression affects 30% of survivors and directly slows motor recovery. "
                    "Please mention your mood score to your medical team. Effective treatment is available."
        })

    # Spasticity
    if data["spasticity"] >= 6:
        recs.append({
            "icon": "💪", "priority": "moderate",
            "title": "Spasticity is high — stretch before every exercise",
            "body": "Always stretch the affected limb for 10–15 minutes before exercise (slow, not bouncing). "
                    "A warm shower before stretching reduces muscle tone significantly."
        })

    return recs[:7]


# ─────────────────────────────────────────────
# EXERCISE LIBRARY
# ─────────────────────────────────────────────

EXERCISES = [
    {
        "name": "Seated Ankle Pumps",
        "reps": "10 reps × 3 sets",
        "duration": "5 minutes",
        "difficulty": "Beginner",
        "target": "Ankle dorsiflexion, circulation",
        "instructions": "Sit in a chair with feet flat on the floor. Slowly lift toes toward your shin, hold 2 seconds, then point toes down. Perform slowly and with control. Rest 30 seconds between sets.",
        "icon": "🦶",
        "lottie": "https://lottie.host/4db68bbd-714e-4f9b-a639-a99d1eab3c0a/VxULmQMUq2.json",
    },
    {
        "name": "Knee Slides",
        "reps": "10 reps each leg",
        "duration": "8 minutes",
        "difficulty": "Beginner",
        "target": "Hip and knee flexion",
        "instructions": "Lie on your back on the bed. Slowly slide one heel toward your bottom, bending the knee. Hold for 3 seconds, then slide back. Repeat on both sides. Keep movements slow and controlled.",
        "icon": "🦵",
        "lottie": "https://lottie.host/b2c3a1e4-f5d6-4a7b-8c9d-0e1f2a3b4c5d/example.json",
    },
    {
        "name": "Standing Weight Shifts",
        "reps": "10 reps × 2 sets",
        "duration": "6 minutes",
        "difficulty": "Beginner",
        "target": "Balance, weight-bearing",
        "instructions": "Stand holding a stable surface (table or chair back). Slowly shift your weight onto your affected leg, hold 3 seconds, then shift back. Keep both feet on the floor throughout.",
        "icon": "⚖️",
        "lottie": "https://lottie.host/1a2b3c4d-5e6f-7a8b-9c0d-1e2f3a4b5c6d/example.json",
    },
    {
        "name": "Seated Marching",
        "reps": "20 reps alternating",
        "duration": "5 minutes",
        "difficulty": "Beginner",
        "target": "Hip flexion, coordination",
        "instructions": "Sit upright in a sturdy chair. Slowly lift one knee up, then lower and lift the other — like marching in place. Keep your back straight and breathe steadily.",
        "icon": "🚶",
        "lottie": "https://lottie.host/2b3c4d5e-6f7a-8b9c-0d1e-2f3a4b5c6d7e/example.json",
    },
    {
        "name": "Ankle Circles",
        "reps": "10 circles each direction, each foot",
        "duration": "4 minutes",
        "difficulty": "Beginner",
        "target": "Ankle mobility, spasticity reduction",
        "instructions": "Sit in a chair with feet lifted slightly off the floor. Rotate each ankle slowly in large circles — 10 clockwise, 10 anti-clockwise. Excellent for reducing morning stiffness.",
        "icon": "🔄",
        "lottie": "https://lottie.host/3c4d5e6f-7a8b-9c0d-1e2f-3a4b5c6d7e8f/example.json",
    },
    {
        "name": "Wall Push-Ups",
        "reps": "10 reps × 3 sets",
        "duration": "7 minutes",
        "difficulty": "Intermediate",
        "target": "Upper body strength, shoulder stability",
        "instructions": "Stand arm's length from a wall. Place both hands flat on the wall at shoulder height. Slowly bend elbows to bring chest toward wall, hold 2 seconds, then push back. Focus on the affected arm doing equal work.",
        "icon": "🤸",
        "lottie": "https://lottie.host/4d5e6f7a-8b9c-0d1e-2f3a-4b5c6d7e8f9a/example.json",
    },
    {
        "name": "Calf Raises",
        "reps": "10 reps × 2 sets",
        "duration": "5 minutes",
        "difficulty": "Intermediate",
        "target": "Calf strength, ankle stability",
        "instructions": "Stand holding a chair back for support. Slowly rise up onto your toes, hold 2-3 seconds, then lower slowly. The slow lowering phase is crucial. If the affected leg is too weak, lead with the stronger leg.",
        "icon": "🏋️",
        "lottie": "https://lottie.host/5e6f7a8b-9c0d-1e2f-3a4b-5c6d7e8f9a0b/example.json",
    },
    {
        "name": "Deep Breathing & Relaxation",
        "reps": "5 minutes",
        "duration": "5 minutes",
        "difficulty": "All levels",
        "target": "Nervous system, blood pressure",
        "instructions": "Sit or lie comfortably. Breathe in slowly through nose for 4 counts, hold for 2 counts, breathe out through mouth for 6 counts. This activates the parasympathetic nervous system and helps reduce blood pressure.",
        "icon": "🧘",
        "lottie": "https://lottie.host/6f7a8b9c-0d1e-2f3a-4b5c-6d7e8f9a0b1c/example.json",
    },
]
