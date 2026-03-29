"""
database.py — Stroke Recovery Monitor v2.0
==========================================
Lightweight JSON-based database for patient records, family access,
daily logs, and alert history. No external database needed.

Author: Samuel Oluwakoya
"""

import json
import os
import hashlib
from datetime import datetime, date

DATA_DIR = "data"
PATIENTS_FILE = os.path.join(DATA_DIR, "patients.json")
LOGS_FILE     = os.path.join(DATA_DIR, "logs.json")
ALERTS_FILE   = os.path.join(DATA_DIR, "alerts.json")


def _ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    for f in [PATIENTS_FILE, LOGS_FILE, ALERTS_FILE]:
        if not os.path.exists(f):
            with open(f, "w") as fh:
                json.dump({}, fh)


def _read(path):
    _ensure_dirs()
    with open(path, "r") as f:
        return json.load(f)


def _write(path, data):
    _ensure_dirs()
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _hash(pin: str) -> str:
    return hashlib.sha256(pin.encode()).hexdigest()


# ─────────────────────────────────────────────
# PATIENT MANAGEMENT
# ─────────────────────────────────────────────

def register_patient(username: str, patient_name: str, age: int,
                     pin: str, family_code: str,
                     family_emails: list, stroke_date: str,
                     affected_side: str, stroke_type: str) -> bool:
    """Register a new patient. Returns False if username already exists."""
    patients = _read(PATIENTS_FILE)
    if username.lower() in patients:
        return False
    patients[username.lower()] = {
        "username":      username.lower(),
        "patient_name":  patient_name,
        "age":           age,
        "pin_hash":      _hash(pin),
        "family_code_hash": _hash(family_code),
        "family_emails": family_emails,
        "stroke_date":   stroke_date,
        "affected_side": affected_side,
        "stroke_type":   stroke_type,
        "registered_on": str(date.today()),
    }
    _write(PATIENTS_FILE, patients)
    return True


def authenticate_patient(username: str, pin: str) -> dict | None:
    """Returns patient dict if credentials are correct, else None."""
    patients = _read(PATIENTS_FILE)
    p = patients.get(username.lower())
    if p and p["pin_hash"] == _hash(pin):
        return p
    return None


def authenticate_family(username: str, family_code: str) -> dict | None:
    """Returns patient dict if family code is correct, else None."""
    patients = _read(PATIENTS_FILE)
    p = patients.get(username.lower())
    if p and p["family_code_hash"] == _hash(family_code):
        return p
    return None


def get_patient(username: str) -> dict | None:
    patients = _read(PATIENTS_FILE)
    return patients.get(username.lower())


def update_patient_field(username: str, field: str, value):
    patients = _read(PATIENTS_FILE)
    if username.lower() in patients:
        patients[username.lower()][field] = value
        _write(PATIENTS_FILE, patients)


# ─────────────────────────────────────────────
# DAILY LOG MANAGEMENT
# ─────────────────────────────────────────────

def save_log_entry(username: str, entry: dict):
    """Save or overwrite today's log entry for this patient."""
    logs = _read(LOGS_FILE)
    key  = username.lower()
    if key not in logs:
        logs[key] = []
    # Replace if same date, else append
    today = str(date.today())
    logs[key] = [e for e in logs[key] if e.get("date") != today]
    entry["date"] = today
    entry["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logs[key].append(entry)
    _write(LOGS_FILE, logs)


def get_logs(username: str) -> list:
    logs = _read(LOGS_FILE)
    return logs.get(username.lower(), [])


def get_today_log(username: str) -> dict | None:
    logs = get_logs(username)
    today = str(date.today())
    for entry in logs:
        if entry.get("date") == today:
            return entry
    return None


# ─────────────────────────────────────────────
# ALERT MANAGEMENT
# ─────────────────────────────────────────────

def save_alert(username: str, alert_type: str, message: str, value=None):
    alerts = _read(ALERTS_FILE)
    key = username.lower()
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
    alerts = _read(ALERTS_FILE)
    all_alerts = alerts.get(username.lower(), [])
    if unread_only:
        return [a for a in all_alerts if not a.get("read")]
    return all_alerts


def mark_alerts_read(username: str):
    alerts = _read(ALERTS_FILE)
    key = username.lower()
    if key in alerts:
        for a in alerts[key]:
            a["read"] = True
        _write(ALERTS_FILE, alerts)


def get_unread_count(username: str) -> int:
    return len(get_alerts(username, unread_only=True))
