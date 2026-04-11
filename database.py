"""
database.py — Stroke Recovery Monitor v2.0
==========================================
Lightweight JSON-based persistence layer.

Uses /tmp/stroke_data/ so data survives page reruns and browser refreshes
on Streamlit Community Cloud. Data resets on full redeployment, which is
acceptable for a research-stage application.

Author: Samuel Oluwakoya
"""

import json
import os
import hashlib
from datetime import datetime, date

# /tmp/ survives Streamlit reruns and stays warm between user sessions
# on Streamlit Cloud — unlike the app working directory which can reset.
DATA_DIR      = "/tmp/stroke_data"
PATIENTS_FILE = os.path.join(DATA_DIR, "patients.json")
LOGS_FILE     = os.path.join(DATA_DIR, "logs.json")
ALERTS_FILE   = os.path.join(DATA_DIR, "alerts.json")


def _ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    for f in [PATIENTS_FILE, LOGS_FILE, ALERTS_FILE]:
        if not os.path.exists(f):
            with open(f, "w") as fh:
                json.dump({}, fh)


def _read(path: str) -> dict:
    _ensure_dirs()
    with open(path, "r") as f:
        return json.load(f)


def _write(path: str, data: dict):
    _ensure_dirs()
    # Write to temp then rename — prevents corruption if process dies mid-write
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _hash(value: str) -> str:
    """SHA-256 hash. Never store plain-text credentials."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────
# PATIENT MANAGEMENT
# ─────────────────────────────────────────────

def register_patient(
    username: str,
    patient_name: str,
    age: int,
    pin: str,
    family_code: str,
    family_emails: list,
    stroke_date: str,
    affected_side: str,
    stroke_type: str,
) -> bool:
    """Register a new patient. Returns False if username already taken."""
    patients = _read(PATIENTS_FILE)
    key = username.strip().lower()
    if key in patients:
        return False
    patients[key] = {
        "username":           key,
        "patient_name":       patient_name.strip(),
        "age":                int(age),
        "pin_hash":           _hash(pin),
        "family_code_hash":   _hash(family_code),
        "family_emails":      [e.strip() for e in family_emails if e.strip()],
        "stroke_date":        stroke_date,
        "affected_side":      affected_side,
        "stroke_type":        stroke_type,
        "registered_on":      str(date.today()),
    }
    _write(PATIENTS_FILE, patients)
    return True


def authenticate_patient(username: str, pin: str) -> dict | None:
    """Returns patient dict on success, None on failure."""
    patients = _read(PATIENTS_FILE)
    p = patients.get(username.strip().lower())
    if p and p.get("pin_hash") == _hash(pin):
        return p
    return None


def authenticate_family(username: str, family_code: str) -> dict | None:
    """Returns patient dict if family code matches, None otherwise."""
    patients = _read(PATIENTS_FILE)
    p = patients.get(username.strip().lower())
    if p and p.get("family_code_hash") == _hash(family_code):
        return p
    return None


def get_patient(username: str) -> dict | None:
    return _read(PATIENTS_FILE).get(username.strip().lower())


def update_patient_field(username: str, field: str, value) -> None:
    patients = _read(PATIENTS_FILE)
    key = username.strip().lower()
    if key in patients:
        patients[key][field] = value
        _write(PATIENTS_FILE, patients)


# ─────────────────────────────────────────────
# DAILY LOG MANAGEMENT
# ─────────────────────────────────────────────

def save_log_entry(username: str, entry: dict) -> None:
    """Save today's log. Replaces any earlier entry for the same date."""
    logs  = _read(LOGS_FILE)
    key   = username.strip().lower()
    today = str(date.today())
    if key not in logs:
        logs[key] = []
    # Remove any existing entry for today before appending the new one
    logs[key] = [e for e in logs[key] if e.get("date") != today]
    entry["date"]      = today
    entry["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logs[key].append(entry)
    _write(LOGS_FILE, logs)


def get_logs(username: str) -> list:
    return _read(LOGS_FILE).get(username.strip().lower(), [])


def get_today_log(username: str) -> dict | None:
    today = str(date.today())
    for entry in get_logs(username):
        if entry.get("date") == today:
            return entry
    return None


# ─────────────────────────────────────────────
# ALERT MANAGEMENT
# ─────────────────────────────────────────────

def save_alert(
    username: str,
    alert_type: str,
    message: str,
    value=None,
) -> None:
    alerts = _read(ALERTS_FILE)
    key    = username.strip().lower()
    if key not in alerts:
        alerts[key] = []
    alerts[key].append({
        "type":      alert_type,
        "message":   message,
        "value":     value,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "date":      str(date.today()),
        "read":      False,
    })
    _write(ALERTS_FILE, alerts)


def get_alerts(username: str, unread_only: bool = False) -> list:
    all_alerts = _read(ALERTS_FILE).get(username.strip().lower(), [])
    if unread_only:
        return [a for a in all_alerts if not a.get("read")]
    return all_alerts


def get_unread_count(username: str) -> int:
    return len(get_alerts(username, unread_only=True))


def mark_alerts_read(username: str) -> None:
    alerts = _read(ALERTS_FILE)
    key    = username.strip().lower()
    if key in alerts:
        for a in alerts[key]:
            a["read"] = True
        _write(ALERTS_FILE, alerts)
