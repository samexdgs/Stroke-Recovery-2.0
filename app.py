"""
Stroke Recovery Monitor v2.0
==============================
Multi-user family monitoring system for stroke recovery.
Patient logs daily data → family receives real-time alerts and dashboard.

Author: Samuel Oluwakoya
GitHub: https://github.com/samueloluwakoya/stroke-recovery-monitor
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import plotly.graph_objects as go
import plotly.express as px

from database import (
    register_patient, authenticate_patient, authenticate_family,
    get_patient, save_log_entry, get_logs, get_today_log,
    save_alert, get_alerts, get_unread_count, mark_alerts_read,
    update_patient_field
)
from ml_engine import (
    load_models, predict, get_recommendations,
    STATUS_LABELS, STATUS_COLORS, STATUS_BG,
    check_bp_alert, EXERCISES
)
from alerts import send_alert_email, build_daily_alert_details


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Stroke Recovery Monitor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ─────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.app-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem; color: #1a1a2e; line-height: 1.2; margin-bottom: 0.2rem;
}
.app-sub { font-size: 0.95rem; color: #6b7280; margin-bottom: 1.5rem; }

.status-card {
    border-radius: 16px; padding: 1.4rem 1.6rem;
    margin-bottom: 1rem; border: 1px solid rgba(0,0,0,0.06);
}
.status-on-track  { background: #d1fae5; border-left: 5px solid #059669; }
.status-plateau   { background: #fef3c7; border-left: 5px solid #d97706; }
.status-attention { background: #fee2e2; border-left: 5px solid #dc2626; }
.status-title { font-size: 1.25rem; font-weight: 600; margin-bottom: 0.3rem; }
.status-desc  { font-size: 0.92rem; color: #374151; }

.metric-box {
    background: #f9fafb; border: 1px solid #e5e7eb;
    border-radius: 12px; padding: 1rem 1.1rem; text-align: center;
}
.metric-label { font-size: 0.75rem; color: #6b7280; text-transform: uppercase;
                letter-spacing: 0.05em; margin-bottom: 0.25rem; }
.metric-value { font-size: 1.7rem; font-weight: 600; color: #111827; }
.metric-unit  { font-size: 0.72rem; color: #9ca3af; }

.metric-bp-ok      { border-top: 4px solid #059669; }
.metric-bp-high    { border-top: 4px solid #dc2626; background: #fff5f5; }
.metric-bp-low     { border-top: 4px solid #2563eb; background: #eff6ff; }
.metric-bp-critical{ border-top: 4px solid #7f1d1d; background: #fee2e2; }

.rec-card {
    background: white; border: 1px solid #e5e7eb;
    border-radius: 12px; padding: 1rem 1.2rem;
    margin-bottom: 0.6rem; border-left: 4px solid #4f46e5;
}
.rec-card-urgent { border-left-color: #dc2626; }
.rec-card-good   { border-left-color: #059669; }
.rec-title { font-weight: 600; color: #1f2937; font-size: 0.93rem; margin-bottom: 0.2rem; }
.rec-body  { color: #4b5563; font-size: 0.86rem; line-height: 1.65; }

.ex-card {
    background: white; border: 1px solid #e5e7eb; border-radius: 14px;
    padding: 1.2rem 1.4rem; margin-bottom: 0.7rem;
}
.ex-title { font-weight: 600; font-size: 1rem; color: #1f2937; margin-bottom: 0.4rem; }
.ex-meta  { font-size: 0.82rem; color: #6b7280; margin-bottom: 0.6rem; }
.ex-instructions { font-size: 0.88rem; color: #374151; line-height: 1.7; }

.alert-item {
    background: #fffbeb; border: 1px solid #fde68a; border-radius: 10px;
    padding: 0.8rem 1rem; margin-bottom: 0.5rem; font-size: 0.88rem;
}
.alert-critical { background: #fff1f2; border-color: #fecaca; }
.alert-time { font-size: 0.78rem; color: #9ca3af; margin-top: 4px; }

.log-row {
    background: #f9fafb; border-radius: 10px; padding: 0.75rem 1rem;
    margin-bottom: 0.4rem; font-size: 0.86rem; border: 1px solid #f3f4f6;
    display: flex; justify-content: space-between; align-items: center;
}

.family-badge {
    background: #ede9fe; color: #5b21b6; border-radius: 8px;
    padding: 4px 10px; font-size: 0.8rem; font-weight: 600;
    display: inline-block; margin-bottom: 8px;
}
.patient-badge {
    background: #d1fae5; color: #065f46; border-radius: 8px;
    padding: 4px 10px; font-size: 0.8rem; font-weight: 600;
    display: inline-block; margin-bottom: 8px;
}

.disclaimer {
    background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 10px;
    padding: 0.8rem 1rem; font-size: 0.8rem; color: #0369a1; margin-top: 1rem;
}

[data-testid="stSidebar"] { background-color: #f8f7ff; }

.stButton > button {
    background: #4f46e5; color: white; border: none;
    border-radius: 10px; padding: 0.55rem 1.5rem;
    font-weight: 500; font-size: 0.92rem; width: 100%;
}
.stButton > button:hover { background: #4338ca; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE HELPERS
# ─────────────────────────────────────────────
def init_session():
    defaults = {
        "logged_in":   False,
        "role":        None,       # "patient" or "family"
        "username":    None,
        "patient":     None,
        "page":        "login",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def logout():
    for k in ["logged_in", "role", "username", "patient", "page"]:
        st.session_state[k] = None
    st.session_state["logged_in"] = False
    st.session_state["page"] = "login"
    st.rerun()


# ─────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────
def radar_chart(data: dict):
    cats   = ["Mobility", "Balance", "Sleep", "Mood", "Exercises", "Low Pain", "Low Fatigue"]
    vals   = [
        data.get("mobility", 5),
        data.get("balance", 5),
        data.get("sleep_quality", 5),
        data.get("mood", 5),
        data.get("exercises_done", 5),
        10 - data.get("pain", 5),
        10 - data.get("fatigue", 5),
    ]
    vals  += vals[:1]
    cats  += cats[:1]
    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=cats, fill="toself",
        fillcolor="rgba(79,70,229,0.15)",
        line=dict(color="#4f46e5", width=2),
        marker=dict(size=5, color="#4f46e5")
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10], tickfont=dict(size=9))),
        showlegend=False,
        margin=dict(l=30, r=30, t=20, b=20),
        height=280,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def trend_chart(logs: list):
    if len(logs) < 2:
        return None
    df = pd.DataFrame(logs)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").tail(30)
    fig = go.Figure()
    series = [
        ("mobility",   "#4f46e5", "Mobility"),
        ("balance",    "#059669", "Balance"),
        ("pain",       "#dc2626", "Pain"),
        ("fatigue",    "#d97706", "Fatigue"),
        ("exercises_done", "#0891b2", "Exercises"),
    ]
    for col, color, label in series:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df["date"], y=df[col], name=label,
                line=dict(color=color, width=2),
                mode="lines+markers", marker=dict(size=5)
            ))
    fig.update_layout(
        height=300, margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", y=-0.2, font=dict(size=11)),
        xaxis=dict(showgrid=True, gridcolor="#f3f4f6"),
        yaxis=dict(title="Score", range=[0, 11], showgrid=True, gridcolor="#f3f4f6"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified"
    )
    return fig


def bp_chart(logs: list):
    if len(logs) < 2:
        return None
    df = pd.DataFrame(logs)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").tail(30)
    if "bp_systolic" not in df.columns:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["bp_systolic"], name="Systolic",
        line=dict(color="#dc2626", width=2), mode="lines+markers"
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["bp_diastolic"], name="Diastolic",
        line=dict(color="#2563eb", width=2), mode="lines+markers"
    ))
    # Normal range bands
    fig.add_hrect(y0=90, y1=120, fillcolor="#d1fae5", opacity=0.25, line_width=0,
                  annotation_text="Normal systolic range", annotation_position="left")
    fig.add_hline(y=140, line_dash="dash", line_color="#d97706",
                  annotation_text="High threshold (140)")
    fig.update_layout(
        height=280, margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", y=-0.2),
        xaxis=dict(showgrid=True, gridcolor="#f3f4f6"),
        yaxis=dict(title="mmHg", range=[50, 220], showgrid=True, gridcolor="#f3f4f6"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def status_timeline(logs: list):
    if len(logs) < 2:
        return None
    df = pd.DataFrame(logs)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").tail(30)
    if "prediction" not in df.columns:
        return None
    label_map = {0: "Needs Attention", 1: "Plateauing", 2: "On Track"}
    color_map  = {0: "#dc2626", 1: "#d97706", 2: "#059669"}
    df["label"] = df["prediction"].map(label_map)
    df["color"] = df["prediction"].map(color_map)
    fig = go.Figure(go.Scatter(
        x=df["date"], y=df["prediction"],
        mode="lines+markers",
        marker=dict(size=12, color=df["color"].tolist()),
        line=dict(color="#d1d5db", width=1.5, dash="dot"),
        text=df["label"],
        hovertemplate="%{x|%b %d}<br>%{text}<extra></extra>"
    ))
    fig.update_layout(
        height=180, margin=dict(l=10, r=10, t=15, b=10),
        yaxis=dict(tickvals=[0, 1, 2],
                   ticktext=["Needs Attention", "Plateauing", "On Track"],
                   showgrid=False),
        xaxis=dict(showgrid=True, gridcolor="#f3f4f6"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def model_bar_chart(model_results: dict):
    names = list(model_results.keys())
    accs  = [model_results[n]["accuracy"] for n in names]
    aucs  = [model_results[n]["roc_auc"]  for n in names]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Accuracy", x=names, y=accs,
                         marker_color="#4f46e5",
                         text=[f"{v:.1%}" for v in accs], textposition="outside"))
    fig.add_trace(go.Bar(name="ROC-AUC",  x=names, y=aucs,
                         marker_color="#059669",
                         text=[f"{v:.3f}" for v in aucs], textposition="outside"))
    fig.update_layout(
        barmode="group", height=260, margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", y=-0.25),
        yaxis=dict(range=[0, 1.15], showgrid=True, gridcolor="#f3f4f6"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


# ─────────────────────────────────────────────
# DAYS SINCE STROKE HELPER
# ─────────────────────────────────────────────
def days_since(stroke_date_str: str) -> int:
    try:
        sd = datetime.strptime(stroke_date_str, "%Y-%m-%d").date()
        return (date.today() - sd).days
    except Exception:
        return 0


# ─────────────────────────────────────────────
# LOGIN PAGE
# ─────────────────────────────────────────────
def page_login():
    col_center = st.columns([1, 2, 1])[1]
    with col_center:
        st.markdown('<div class="app-title">🧠 Stroke Recovery Monitor</div>', unsafe_allow_html=True)
        st.markdown('<div class="app-sub">Family-connected recovery tracking · Powered by ML</div>', unsafe_allow_html=True)
        st.markdown("---")

        mode = st.radio("I am:", ["Patient (Mama logs in here)", "Family Member (monitoring dashboard)"],
                        horizontal=True)

        st.markdown("#### Sign In")
        username = st.text_input("Patient username", placeholder="e.g. mama_grace").strip().lower()

        if "Patient" in mode:
            pin = st.text_input("Your PIN", type="password", placeholder="4-digit PIN")
            if st.button("Sign In as Patient"):
                if not username or not pin:
                    st.error("Please enter username and PIN.")
                else:
                    patient = authenticate_patient(username, pin)
                    if patient:
                        st.session_state["logged_in"] = True
                        st.session_state["role"]      = "patient"
                        st.session_state["username"]  = username
                        st.session_state["patient"]   = patient
                        st.session_state["page"]      = "patient_dashboard"
                        st.rerun()
                    else:
                        st.error("Incorrect username or PIN.")
        else:
            family_code = st.text_input("Family access code", type="password",
                                        placeholder="Code given to family by the patient's carer")
            if st.button("Sign In as Family"):
                if not username or not family_code:
                    st.error("Please enter the patient username and family code.")
                else:
                    patient = authenticate_family(username, family_code)
                    if patient:
                        st.session_state["logged_in"] = True
                        st.session_state["role"]      = "family"
                        st.session_state["username"]  = username
                        st.session_state["patient"]   = patient
                        st.session_state["page"]      = "family_dashboard"
                        st.rerun()
                    else:
                        st.error("Incorrect username or family access code.")

        st.markdown("---")
        with st.expander("New patient? Register here"):
            page_register()


# ─────────────────────────────────────────────
# REGISTRATION PAGE
# ─────────────────────────────────────────────
def page_register():
    st.markdown("#### Register a New Patient")
    st.info("This is usually done by the patient's carer or family coordinator once.")

    r_name     = st.text_input("Patient's full name", key="r_name")
    r_username = st.text_input("Choose a username (no spaces)", key="r_username",
                               placeholder="mama_grace").strip().lower()
    r_age      = st.number_input("Patient's age", min_value=18, max_value=110,
                                  value=65, key="r_age")
    r_stroke   = st.date_input("Date of stroke", key="r_stroke",
                                value=date.today() - timedelta(days=30),
                                max_value=date.today())
    r_side     = st.selectbox("Affected side", ["Left", "Right"], key="r_side")
    r_type     = st.selectbox("Stroke type", ["Ischemic", "Hemorrhagic", "Unknown"], key="r_type")

    st.markdown("**Security**")
    r_pin        = st.text_input("Patient PIN (4 digits)", type="password", key="r_pin",
                                  placeholder="Patient uses this to log in daily")
    r_pin2       = st.text_input("Confirm PIN", type="password", key="r_pin2")
    r_fam_code   = st.text_input("Family access code", type="password", key="r_fam",
                                  placeholder="Family members use this to view the dashboard")
    r_fam_code2  = st.text_input("Confirm family code", type="password", key="r_fam2")

    st.markdown("**Family alert emails** *(optional — enter one per line)*")
    r_emails_raw = st.text_area("Family email addresses", key="r_emails",
                                 placeholder="daughter@email.com\nson@email.com\ndoctor@hospital.com",
                                 height=90)

    if st.button("Create Patient Profile", key="reg_btn"):
        errors = []
        if not r_name:       errors.append("Patient name is required.")
        if not r_username:   errors.append("Username is required.")
        if " " in r_username: errors.append("Username cannot contain spaces.")
        if not r_pin:        errors.append("PIN is required.")
        if r_pin != r_pin2:  errors.append("PINs do not match.")
        if not r_fam_code:   errors.append("Family access code is required.")
        if r_fam_code != r_fam_code2: errors.append("Family codes do not match.")

        if errors:
            for e in errors:
                st.error(e)
        else:
            emails = [e.strip() for e in r_emails_raw.splitlines() if e.strip()]
            success = register_patient(
                username=r_username,
                patient_name=r_name,
                age=int(r_age),
                pin=r_pin,
                family_code=r_fam_code,
                family_emails=emails,
                stroke_date=str(r_stroke),
                affected_side=r_side,
                stroke_type=r_type,
            )
            if success:
                st.success(f"✅ Profile created for **{r_name}**! "
                           f"Username: `{r_username}`. They can now sign in.")
            else:
                st.error(f"Username `{r_username}` is already taken. Please choose another.")


# ─────────────────────────────────────────────
# PATIENT DASHBOARD
# ─────────────────────────────────────────────
def page_patient_dashboard(model_results, best_name):
    patient  = st.session_state["patient"]
    username = st.session_state["username"]
    name     = patient["patient_name"]
    logs     = get_logs(username)
    today_log = get_today_log(username)
    days_post = days_since(patient.get("stroke_date", str(date.today())))

    # ── Sidebar ──
    with st.sidebar:
        st.markdown(f'<div class="patient-badge">👤 Patient</div>', unsafe_allow_html=True)
        st.markdown(f"### Hello, {name.split()[0]}! 👋")
        st.markdown(f"*{date.today().strftime('%A, %d %B %Y')}*")
        st.markdown(f"**Day {days_post}** of your recovery journey")
        st.markdown("---")
        st.markdown("**Navigation**")
        nav = st.radio("", ["📋 Daily Check-In", "📈 My Progress", "🏃 Exercise Guide", "📊 Model Info"],
                       label_visibility="collapsed")
        st.markdown("---")
        if st.button("🔓 Sign Out"):
            logout()

    st.markdown(f'<div class="app-title">🧠 Recovery Dashboard</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="app-sub">Welcome back, {name} · Day {days_post} of recovery</div>',
                unsafe_allow_html=True)

    # ─────────────────────────────────────────
    if nav == "📋 Daily Check-In":
        if today_log:
            st.success(f"✅ You have already completed today's check-in at {today_log.get('timestamp','')[:16]}. "
                       "Great job! See your results below or come back tomorrow.")
            _render_results(today_log, patient)
        else:
            _render_checkin_form(patient, username, model_results, best_name, days_post)

    elif nav == "📈 My Progress":
        _render_patient_progress(logs, name)

    elif nav == "🏃 Exercise Guide":
        _render_exercise_guide()

    elif nav == "📊 Model Info":
        _render_model_info(model_results, best_name)


# ── Daily check-in form ──
def _render_checkin_form(patient, username, model_results, best_name, days_post):
    st.markdown("### 📋 Today's Check-In")
    st.markdown("*Fill in how you are feeling right now. This takes about 2 minutes.*")

    with st.form("daily_checkin"):
        st.markdown("#### 💉 Blood Pressure *(ask your nurse or use home monitor)*")
        col1, col2 = st.columns(2)
        with col1:
            bp_sys  = st.number_input("Systolic (top number)", min_value=60, max_value=250, value=125)
        with col2:
            bp_dia  = st.number_input("Diastolic (bottom number)", min_value=40, max_value=150, value=80)

        st.markdown("#### 😔 How are you feeling?")
        col1, col2 = st.columns(2)
        with col1:
            pain     = st.slider("Pain level",          1, 10, 3, help="1 = no pain · 10 = worst pain")
            fatigue  = st.slider("Fatigue / tiredness", 1, 10, 4, help="1 = full of energy · 10 = exhausted")
            spastic  = st.slider("Stiffness in limbs",  1, 10, 3, help="1 = no stiffness · 10 = very stiff")
            sleep    = st.slider("Sleep quality",       1, 10, 6, help="1 = terrible night · 10 = excellent")
        with col2:
            balance  = st.slider("Balance",             1, 10, 6, help="1 = very unsteady · 10 = very steady")
            mobility = st.slider("Walking ability",     1, 10, 6, help="1 = cannot walk · 10 = walks normally")
            mood     = st.slider("Mood today",          1, 10, 7, help="1 = very low · 10 = very positive")

        st.markdown("#### 🏃 Exercises")
        col1, col2 = st.columns(2)
        with col1:
            ex_done = st.slider("Exercises completed (out of 10)", 0, 10, 5)
        with col2:
            ex_min  = st.slider("Total exercise time (minutes)",  0, 90,  30)

        st.markdown("#### 🦿 Assistive Devices Used Today")
        col1, col2, col3 = st.columns(3)
        with col1:
            afo    = st.checkbox("Wearing AFO")
        with col2:
            cane   = st.checkbox("Using cane")
        with col3:
            walker = st.checkbox("Using walker")

        st.markdown("#### 💬 How was your day? *(optional)*")
        notes = st.text_area("Any notes for your family or doctor", placeholder="e.g. 'Felt stronger today, left knee was stiff in the morning...'", height=80)

        submitted = st.form_submit_button("📊 Submit & Analyse My Recovery")

    if submitted:
        data = {
            "age":           patient["age"],
            "days_post":     days_post,
            "affected_side": 1 if patient.get("affected_side") == "Right" else 0,
            "stroke_type":   1 if patient.get("stroke_type")   == "Hemorrhagic" else 0,
            "pain":          pain,
            "fatigue":       fatigue,
            "spasticity":    spastic,
            "balance":       balance,
            "mobility":      mobility,
            "exercises_done":ex_done,
            "exercise_min":  ex_min,
            "sleep_quality": sleep,
            "mood":          mood,
            "uses_afo":      int(afo),
            "uses_cane":     int(cane),
            "uses_walker":   int(walker),
            "bp_systolic":   bp_sys,
            "bp_diastolic":  bp_dia,
            "notes":         notes,
        }

        pred, proba = predict(model_results, best_name, data)
        log_entry   = {**data, "prediction": pred, "proba": proba}
        save_log_entry(username, log_entry)

        # ── Trigger alerts ──
        is_bp_alert, bp_severity, bp_msg = check_bp_alert(bp_sys, bp_dia)

        if is_bp_alert:
            alert_type = "bp_critical" if bp_severity == "critical" else f"bp_{bp_severity}"
            save_alert(username, alert_type, bp_msg, {"systolic": bp_sys, "diastolic": bp_dia})
            send_alert_email(
                patient_name=patient["patient_name"],
                family_emails=patient.get("family_emails", []),
                alert_type=alert_type,
                message=bp_msg,
                details={"Blood Pressure": f"{bp_sys}/{bp_dia} mmHg",
                         "Severity": bp_severity.upper(),
                         "Time": datetime.now().strftime("%I:%M %p")}
            )

        if pred == 0:
            save_alert(username, "needs_attention",
                       f"{patient['patient_name']}'s recovery needs clinical attention today.")
        
        # Always send daily completion alert
        details = build_daily_alert_details(data, pred, proba)
        send_alert_email(
            patient_name=patient["patient_name"],
            family_emails=patient.get("family_emails", []),
            alert_type="daily_complete",
            message=f"{patient['patient_name']} has completed their daily recovery check-in.",
            details=details
        )

        # Save and show results
        today_log = get_today_log(username)
        st.rerun()


# ── Results display (after submission or returning) ──
def _render_results(today_log: dict, patient: dict):
    pred  = today_log.get("prediction", 1)
    proba = today_log.get("proba", [0.2, 0.5, 0.3])

    status_class = {0: "status-attention", 1: "status-plateau", 2: "status-on-track"}
    status_desc  = {
        0: "Your metrics today suggest you need clinical support. Please contact your physiotherapist.",
        1: "You're in a plateau — common and manageable. Try a new exercise or environment this week.",
        2: "Excellent. Your metrics today show solid recovery momentum. Keep going!"
    }

    st.markdown(f"""
    <div class="status-card {status_class[pred]}">
        <div class="status-title">{STATUS_LABELS[pred]}</div>
        <div class="status-desc">{status_desc[pred]}</div>
    </div>
    """, unsafe_allow_html=True)

    # Confidence
    c1, c2, c3 = st.columns(3)
    labels = ["Needs Attention", "Plateauing", "On Track"]
    for col, label, prob in zip([c1, c2, c3], labels, proba):
        col.metric(label, f"{prob:.0%}")

    st.markdown("---")

    # BP card
    bp_sys = today_log.get("bp_systolic", 0)
    bp_dia = today_log.get("bp_diastolic", 0)
    is_alert, bp_sev, _ = check_bp_alert(bp_sys, bp_dia)
    bp_class = {"normal": "metric-bp-ok", "high": "metric-bp-high",
                "low": "metric-bp-low", "critical": "metric-bp-critical"}.get(bp_sev, "metric-bp-ok")

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.markdown(f"""<div class="metric-box {bp_class}">
        <div class="metric-label">Blood Pressure</div>
        <div class="metric-value" style="font-size:1.3rem;">{bp_sys}/{bp_dia}</div>
        <div class="metric-unit">mmHg</div></div>""", unsafe_allow_html=True)
    
    for col, label, key, unit in [
        (col2, "Mobility",   "mobility",        "/10"),
        (col3, "Balance",    "balance",         "/10"),
        (col4, "Exercises",  "exercises_done",  "/10"),
        (col5, "Pain",       "pain",            "/10"),
        (col6, "Sleep",      "sleep_quality",   "/10"),
    ]:
        col.markdown(f"""<div class="metric-box">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{today_log.get(key, '—')}</div>
            <div class="metric-unit">{unit}</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_r, col_radar = st.columns([1.4, 1])

    with col_r:
        st.markdown("**Personalised Recommendations**")
        recs = get_recommendations(pred, today_log)
        for r in recs:
            card_class = "rec-card-urgent" if r["priority"] == "urgent" else (
                         "rec-card-good"   if r["priority"] == "good"   else "")
            st.markdown(f"""
            <div class="rec-card {card_class}">
                <div class="rec-title">{r['icon']} {r['title']}</div>
                <div class="rec-body">{r['body']}</div>
            </div>""", unsafe_allow_html=True)

    with col_radar:
        st.markdown("**Today's Recovery Profile**")
        st.plotly_chart(radar_chart(today_log), use_container_width=True)

    if today_log.get("notes"):
        st.markdown(f"**Your notes:** *{today_log['notes']}*")

    st.markdown("""<div class="disclaimer">
    ⚕️ <strong>Medical Disclaimer:</strong> This tool provides informational support only.
    It does not constitute medical advice. Always consult your qualified rehabilitation team.
    </div>""", unsafe_allow_html=True)


def _render_patient_progress(logs: list, name: str):
    st.markdown("### 📈 My Recovery Progress")
    if len(logs) < 2:
        st.info("Log at least **2 days** of data to see progress charts.")
        return

    total_days  = len(logs)
    on_track    = sum(1 for e in logs if e.get("prediction") == 2)
    streak      = 0
    for e in reversed(logs):
        if e.get("prediction") == 2:
            streak += 1
        else:
            break

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Days logged", total_days)
    c2.metric("Days on track", on_track)
    c3.metric("Best streak 🔥", streak)
    c4.metric("Consistency", f"{on_track/total_days:.0%}")

    st.markdown("**Recovery Status Timeline**")
    fig = status_timeline(logs)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Daily Metric Trends**")
        fig2 = trend_chart(logs)
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)
    with col_b:
        st.markdown("**Blood Pressure History**")
        fig3 = bp_chart(logs)
        if fig3:
            st.plotly_chart(fig3, use_container_width=True)

    with st.expander("View all my data"):
        df = pd.DataFrame(logs)
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "⬇️ Download my data (CSV)",
            data=df.to_csv(index=False),
            file_name=f"recovery_log_{date.today()}.csv",
            mime="text/csv"
        )


def _exercise_card_html(ex: dict, diff_color: str) -> str:
    """
    Renders a complete exercise card as a single self-contained HTML block.
    The Lottie player is loaded via the official @lottiefiles/lottie-player
    script tag — the most widely compatible approach.
    Falls back to a CSS-animated SVG figure if the animation fails or URL is empty.
    Everything is in one components.html() call — no st.markdown mixing.
    """
    lottie_url = ex.get("lottie", "")
    icon       = ex.get("icon", "🏃")
    name       = ex["name"]
    reps       = ex["reps"]
    duration   = ex["duration"]
    difficulty = ex["difficulty"]
    target     = ex["target"]
    instructions = ex["instructions"]

    # CSS-animated SVG fallback — a simple stick figure that bounces
    # This always works, no network needed
    fallback_svg = f"""
    <div id="fallback-{name.replace(' ','-')}"
         style="display:flex;flex-direction:column;align-items:center;
                justify-content:center;height:240px;background:#f0f4ff;
                border-radius:14px;">
        <svg width="80" height="120" viewBox="0 0 80 120"
             style="animation:bounce 1.2s ease-in-out infinite;">
            <style>
                @keyframes bounce {{
                    0%,100% {{ transform: translateY(0); }}
                    50%      {{ transform: translateY(-10px); }}
                }}
                @keyframes sway {{
                    0%,100% {{ transform: rotate(-15deg); }}
                    50%     {{ transform: rotate(15deg); }}
                }}
            </style>
            <!-- head -->
            <circle cx="40" cy="14" r="10" fill="#4f46e5" opacity="0.85"/>
            <!-- body -->
            <line x1="40" y1="24" x2="40" y2="70"
                  stroke="#4f46e5" stroke-width="4" stroke-linecap="round"/>
            <!-- arms -->
            <g style="transform-origin:40px 40px;animation:sway 1.2s ease-in-out infinite;">
                <line x1="40" y1="38" x2="15" y2="58"
                      stroke="#4f46e5" stroke-width="3" stroke-linecap="round"/>
                <line x1="40" y1="38" x2="65" y2="58"
                      stroke="#4f46e5" stroke-width="3" stroke-linecap="round"/>
            </g>
            <!-- legs -->
            <line x1="40" y1="70" x2="20" y2="100"
                  stroke="#4f46e5" stroke-width="3" stroke-linecap="round"/>
            <line x1="40" y1="70" x2="60" y2="100"
                  stroke="#4f46e5" stroke-width="3" stroke-linecap="round"/>
        </svg>
        <div style="font-size:1.8rem;margin-top:8px;">{icon}</div>
        <div style="font-size:0.78rem;color:#6b7280;margin-top:4px;">
            Watch and copy this movement
        </div>
    </div>
    """

    if lottie_url:
        animation_block = f"""
        <script src="https://unpkg.com/@lottiefiles/lottie-player@2.0.8/dist/lottie-player.js"></script>
        <div id="lottie-wrap-{name.replace(' ','-')}"
             style="background:#f0f4ff;border-radius:14px;
                    display:flex;align-items:center;justify-content:center;
                    min-height:240px;overflow:hidden;">
            <lottie-player
                src="{lottie_url}"
                background="transparent"
                speed="1"
                style="width:240px;height:240px;"
                loop autoplay
                onerror="
                    document.getElementById('lottie-wrap-{name.replace(' ','-')}').innerHTML =
                    document.getElementById('fb-{name.replace(' ','-')}').innerHTML;
                ">
            </lottie-player>
        </div>
        <div id="fb-{name.replace(' ','-')}" style="display:none;">
            {fallback_svg}
        </div>
        """
    else:
        animation_block = fallback_svg

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        * {{ box-sizing:border-box; margin:0; padding:0; }}
        body {{ font-family: system-ui, -apple-system, sans-serif;
                background:transparent; padding:0; }}
        .card {{
            display:flex; gap:20px; padding:4px 2px;
            align-items:flex-start;
        }}
        .anim-col {{
            flex: 0 0 250px;
        }}
        .info-col {{
            flex:1; min-width:0;
        }}
        .ex-name {{
            font-size:1.05rem; font-weight:600; color:#1f2937;
            margin-bottom:10px;
        }}
        .badges {{
            display:flex; flex-wrap:wrap; gap:6px; margin-bottom:12px;
        }}
        .badge {{
            border-radius:8px; padding:3px 10px; font-size:0.8rem;
            background:#f3f4f6; color:#374151;
        }}
        .badge-diff {{
            font-weight:600; background:#ecfdf5; color:{diff_color};
        }}
        .target {{
            font-size:0.8rem; color:#6b7280; margin-bottom:12px;
        }}
        .instructions-box {{
            background:#f9fafb; border-radius:12px; padding:14px;
            border-left:4px solid #4f46e5;
        }}
        .instructions-label {{
            font-size:0.82rem; font-weight:600; color:#4f46e5;
            margin-bottom:6px;
        }}
        .instructions-text {{
            font-size:0.86rem; color:#374151; line-height:1.75;
        }}
    </style>
    </head>
    <body>
    <div class="card">
        <div class="anim-col">
            {animation_block}
        </div>
        <div class="info-col">
            <div class="ex-name">{icon} {name}</div>
            <div class="badges">
                <span class="badge">⏱ {duration}</span>
                <span class="badge">🔁 {reps}</span>
                <span class="badge badge-diff">{difficulty}</span>
            </div>
            <div class="target">🎯 <strong>Target:</strong> {target}</div>
            <div class="instructions-box">
                <div class="instructions-label">📋 How to do it</div>
                <div class="instructions-text">{instructions}</div>
            </div>
        </div>
    </div>
    </body>
    </html>
    """


def _render_exercise_guide():
    import streamlit.components.v1 as components

    st.markdown("### 🏃 Daily Exercise Guide")
    st.markdown("*Watch each animation, then copy the movement. Stop if pain goes above 7.*")

    diff_colors = {
        "Beginner":    "#059669",
        "Intermediate":"#d97706",
        "All levels":  "#4f46e5"
    }

    for i, ex in enumerate(EXERCISES):
        diff_color = diff_colors.get(ex["difficulty"], "#6b7280")

        with st.expander(
            f"{ex['icon']}  {ex['name']}  —  {ex['reps']}  ·  {ex['difficulty']}",
            expanded=(i == 0)
        ):
            components.html(
                _exercise_card_html(ex, diff_color),
                height=300,
                scrolling=False
            )

    st.markdown("""<div class="disclaimer">
    ⚕️ Always consult your physiotherapist before starting new exercises.
    Stop immediately if you feel chest pain, severe dizziness, or pain above 8/10.
    </div>""", unsafe_allow_html=True)


def _render_model_info(model_results, best_name):
    st.markdown("### 📊 ML Model Performance")
    st.markdown(f"**Active model: `{best_name}`** — selected automatically by highest ROC-AUC")
    st.plotly_chart(model_bar_chart(model_results), use_container_width=True)
    st.markdown("""
    **What these numbers mean:**
    - **Accuracy** — % of test patients correctly classified
    - **ROC-AUC** — ability to distinguish between recovery classes (1.0 = perfect)
    
    **Dataset:** 2,000 simulated stroke survivor records · 18 clinical features · 80/20 train-test split
    """)


# ─────────────────────────────────────────────
# FAMILY DASHBOARD
# ─────────────────────────────────────────────
def page_family_dashboard():
    patient  = st.session_state["patient"]
    username = st.session_state["username"]
    name     = patient["patient_name"]
    logs     = get_logs(username)
    today_log = get_today_log(username)
    alerts   = get_alerts(username)
    unread   = get_unread_count(username)
    days_post = days_since(patient.get("stroke_date", str(date.today())))

    with st.sidebar:
        st.markdown(f'<div class="family-badge">👨‍👩‍👧 Family View</div>', unsafe_allow_html=True)
        st.markdown(f"### Monitoring {name.split()[0]}")
        st.markdown(f"*{date.today().strftime('%A, %d %B %Y')}*")
        st.markdown(f"**Day {days_post}** of recovery")
        st.markdown("---")
        nav = st.radio("", ["🏠 Overview", "📈 Progress Charts", "🔔 Alert History", "📋 Full Log"],
                       label_visibility="collapsed")
        if unread > 0:
            st.warning(f"🔔 {unread} unread alerts")
        st.markdown("---")
        if st.button("🔓 Sign Out"):
            logout()

    st.markdown(f'<div class="app-title">👨‍👩‍👧 Family Monitoring Dashboard</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="app-sub">You are viewing <strong>{name}</strong>\'s recovery · Read-only access</div>',
                unsafe_allow_html=True)

    if nav == "🏠 Overview":
        _render_family_overview(today_log, logs, patient, username, alerts, unread, days_post)
    elif nav == "📈 Progress Charts":
        _render_patient_progress(logs, name)
    elif nav == "🔔 Alert History":
        _render_alert_history(alerts, username)
    elif nav == "📋 Full Log":
        _render_full_log(logs, name)


def _render_family_overview(today_log, logs, patient, username, alerts, unread, days_post):
    name = patient["patient_name"]

    # ── Today's check-in status ──
    if today_log:
        pred  = today_log.get("prediction", 1)
        proba = today_log.get("proba", [0.2, 0.5, 0.3])

        status_class = {0: "status-attention", 1: "status-plateau", 2: "status-on-track"}
        status_msg   = {
            0: f"⚠️ {name}'s recovery metrics need clinical attention today. "
               "Consider calling her doctor.",
            1: f"{name} is in a plateau phase. Her exercises and routine may need adjustment.",
            2: f"{name} is making solid recovery progress today. Keep encouraging her!"
        }

        st.markdown(f"""
        <div class="status-card {status_class[pred]}">
            <div class="status-title">{STATUS_LABELS[pred]}</div>
            <div class="status-desc">{status_msg[pred]}</div>
        </div>""", unsafe_allow_html=True)

        # BP alert
        bp_sys = today_log.get("bp_systolic", 0)
        bp_dia = today_log.get("bp_diastolic", 0)
        is_alert, bp_sev, bp_msg = check_bp_alert(bp_sys, bp_dia)
        if is_alert:
            bp_alert_class = "alert-critical" if bp_sev == "critical" else "alert-item"
            st.markdown(f"""
            <div class="{bp_alert_class}">
                💉 <strong>Blood Pressure Alert</strong> — {bp_msg}
            </div>""", unsafe_allow_html=True)

        # Key metrics
        st.markdown("#### Today's Metrics")
        bp_class = {"normal": "metric-bp-ok", "high": "metric-bp-high",
                    "low": "metric-bp-low", "critical": "metric-bp-critical"}.get(bp_sev, "metric-bp-ok")

        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        col1.markdown(f"""<div class="metric-box {bp_class}">
            <div class="metric-label">Blood Pressure</div>
            <div class="metric-value" style="font-size:1.2rem;">{bp_sys}/{bp_dia}</div>
            <div class="metric-unit">mmHg</div></div>""", unsafe_allow_html=True)

        for col, label, key in [
            (col2, "Mobility",   "mobility"),
            (col3, "Balance",    "balance"),
            (col4, "Exercises",  "exercises_done"),
            (col5, "Pain",       "pain"),
            (col6, "Sleep",      "sleep_quality"),
            (col7, "Mood",       "mood"),
        ]:
            val = today_log.get(key, "—")
            col.markdown(f"""<div class="metric-box">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-unit">/10</div></div>""", unsafe_allow_html=True)

        # Confidence
        st.markdown("#### Recovery Status Confidence")
        c1, c2, c3 = st.columns(3)
        for col, label, prob in zip([c1, c2, c3],
                                    ["Needs Attention", "Plateauing", "On Track"],
                                    proba):
            col.metric(label, f"{prob:.0%}")

        # Radar + recommendations side by side
        col_recs, col_radar = st.columns([1.4, 1])
        with col_recs:
            st.markdown("#### Today's Recommendations for Mama's Carers")
            recs = get_recommendations(pred, today_log)
            for r in recs:
                card_class = "rec-card-urgent" if r["priority"] == "urgent" else (
                             "rec-card-good"   if r["priority"] == "good"   else "")
                st.markdown(f"""
                <div class="rec-card {card_class}">
                    <div class="rec-title">{r['icon']} {r['title']}</div>
                    <div class="rec-body">{r['body']}</div>
                </div>""", unsafe_allow_html=True)
        with col_radar:
            st.markdown("#### Recovery Profile")
            st.plotly_chart(radar_chart(today_log), use_container_width=True)

        if today_log.get("notes"):
            st.info(f"💬 **{name}'s note today:** *{today_log['notes']}*")

        # Unread alerts
        if unread > 0:
            st.markdown(f"#### 🔔 {unread} Unread Alerts")
            for a in alerts[-5:]:
                if not a.get("read"):
                    cls = "alert-critical" if "critical" in a.get("type","") else "alert-item"
                    st.markdown(f"""
                    <div class="{cls}">
                        <strong>{a['message']}</strong>
                        <div class="alert-time">{a['timestamp']}</div>
                    </div>""", unsafe_allow_html=True)
            mark_alerts_read(username)

    else:
        st.warning(f"⏳ **{name} has not logged today yet.**")
        st.markdown("She usually logs between [set a time]. You can send her a reminder.")

        if len(logs) > 0:
            last = logs[-1]
            last_date = last.get("date", "unknown")
            st.markdown(f"Last log: **{last_date}** — "
                        f"Status: {STATUS_LABELS.get(last.get('prediction', 1), 'Unknown')}")

    # Summary strip
    st.markdown("---")
    st.markdown("#### Recovery at a Glance")
    total = len(logs)
    on_track = sum(1 for e in logs if e.get("prediction") == 2)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total days logged", total)
    col2.metric("Days on track", on_track)
    col3.metric("Day of recovery", days_post)
    col4.metric("Total alerts", len(alerts))

    if len(logs) >= 2:
        st.markdown("**30-Day Blood Pressure Trend**")
        fig = bp_chart(logs)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("""<div class="disclaimer">
    👨‍👩‍👧 <strong>Family view is read-only.</strong> This dashboard shows your loved one's self-reported data.
    It does not replace clinical assessment. If you are concerned about her condition, contact her medical team directly.
    </div>""", unsafe_allow_html=True)


def _render_alert_history(alerts: list, username: str):
    st.markdown("### 🔔 Alert History")
    if not alerts:
        st.info("No alerts yet. Alerts are generated when blood pressure is abnormal or recovery needs attention.")
        return

    mark_alerts_read(username)
    for a in reversed(alerts):
        cls = "alert-critical" if "critical" in a.get("type", "") else "alert-item"
        st.markdown(f"""
        <div class="{cls}">
            <strong>{a['type'].replace('_', ' ').title()}</strong> — {a['message']}
            <div class="alert-time">{a['timestamp']}</div>
        </div>""", unsafe_allow_html=True)


def _render_full_log(logs: list, name: str):
    st.markdown(f"### 📋 Full Recovery Log — {name}")
    if not logs:
        st.info("No log entries yet.")
        return
    df = pd.DataFrame(logs)
    if "prediction" in df.columns:
        df["status"] = df["prediction"].map({0:"Needs Attention", 1:"Plateauing", 2:"On Track"})
    st.dataframe(df, use_container_width=True)
    st.download_button(
        "⬇️ Download full log (CSV)",
        data=df.to_csv(index=False),
        file_name=f"{name.replace(' ','_')}_recovery_log_{date.today()}.csv",
        mime="text/csv"
    )


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────
def main():
    init_session()
    model_results, best_name, _ = load_models()

    if not st.session_state["logged_in"]:
        page_login()
    elif st.session_state["role"] == "patient":
        page_patient_dashboard(model_results, best_name)
    elif st.session_state["role"] == "family":
        page_family_dashboard()
    else:
        st.error("Unknown session state. Please sign out and try again.")
        if st.button("Go to login"):
            logout()


if __name__ == "__main__":
    main()
