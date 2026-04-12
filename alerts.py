"""
alerts.py — Stroke Recovery Monitor v2.0
Using Brevo (brevo.com) — free, sends to ANY email, no domain needed.

SETUP for Samuel (once only):
  1. brevo.com → Sign up free
  2. Settings → SMTP & API → API Keys → Create API key → copy it
  3. Streamlit app → Settings → Secrets → add:
       BREVO_API_KEY = "xkeysib-your-key-here"
  Done. Patients just type family emails. Nothing else needed.

Free tier: 300 emails/day to ANY recipient, no domain verification.
"""
import streamlit as st
from datetime import datetime
import urllib.request, urllib.error, json


def _get_api_key():
    try:
        return st.secrets.get("BREVO_API_KEY", None)
    except Exception:
        return None


def send_alert_email(patient_name, family_emails, alert_type, message, details=None):
    api_key = _get_api_key()
    if not api_key:
        st.warning("Email alerts not active. BREVO_API_KEY missing from Streamlit Secrets.", icon="📧")
        return False
    if not family_emails:
        return False

    config = {
        "bp_critical":     ("URGENT: Blood Pressure Emergency",      "#dc2626"),
        "bp_high":         ("Alert: High Blood Pressure Detected",   "#d97706"),
        "bp_low":          ("Alert: Low Blood Pressure",             "#2563eb"),
        "needs_attention": ("Recovery Alert: Needs Attention Today", "#dc2626"),
        "daily_complete":  ("Daily Check-In Completed",              "#059669"),
        "plateau":         ("Recovery Update: Plateau Phase",        "#d97706"),
        "on_track":        ("Recovery Update: On Track",             "#059669"),
    }
    subject, accent = config.get(alert_type, ("Recovery Update", "#4f46e5"))
    timestamp = datetime.now().strftime("%A, %d %B %Y at %I:%M %p")

    details_html = ""
    if details:
        rows = "".join(
            f"<tr><td style='padding:8px 14px;color:#6b7280;font-size:14px;"
            f"border-bottom:1px solid #f3f4f6;white-space:nowrap'>{k}</td>"
            f"<td style='padding:8px 14px;font-weight:600;font-size:14px;"
            f"border-bottom:1px solid #f3f4f6'>{v}</td></tr>"
            for k, v in details.items()
        )
        details_html = (
            f"<table style='width:100%;border-collapse:collapse;margin-top:16px;"
            f"border:1px solid #e5e7eb;border-radius:8px;overflow:hidden;'>"
            f"<tbody>{rows}</tbody></table>"
        )

    html = f"""<!DOCTYPE html><html><body style='margin:0;padding:0;background:#f9fafb;font-family:Arial,sans-serif;'>
  <div style='max-width:560px;margin:30px auto;background:white;border-radius:16px;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,0.08);'>
    <div style='background:{accent};padding:28px 32px;'>
      <div style='font-size:20px;font-weight:700;color:white;'>{subject}</div>
      <div style='font-size:13px;color:rgba(255,255,255,0.85);margin-top:6px;'>Stroke Recovery Monitor &middot; {timestamp}</div>
    </div>
    <div style='padding:28px 32px;'>
      <p style='font-size:15px;color:#374151;line-height:1.6;margin-top:0;'>Automated update for <strong>{patient_name}</strong>.</p>
      <div style='background:#f9fafb;border-left:4px solid {accent};border-radius:0 8px 8px 0;padding:16px 20px;margin:20px 0;'>
        <div style='font-size:15px;color:#1f2937;line-height:1.6;'>{message}</div>
      </div>
      {details_html}
      <p style='font-size:13px;color:#9ca3af;margin-top:24px;line-height:1.6;'>Log in to the Stroke Recovery Monitor app to view the full dashboard.</p>
    </div>
    <div style='background:#f9fafb;padding:18px 32px;border-top:1px solid #e5e7eb;'>
      <div style='font-size:12px;color:#9ca3af;'>Built by Samuel Oluwakoya &middot; Stroke Recovery Monitor v2.0<br>Automated alert.</div>
    </div>
  </div></body></html>"""

    errors = []
    success_count = 0

    for recipient in family_emails:
        payload = json.dumps({
            "sender":      {"name": "Stroke Recovery Monitor", "email": "noreply@strokemonitor.app"},
            "to":          [{"email": recipient}],
            "subject":     subject,
            "htmlContent": html,
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.brevo.com/v3/smtp/email",
            data=payload,
            headers={"accept": "application/json", "api-key": api_key, "content-type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                if resp.status in (200, 201):
                    success_count += 1
                else:
                    errors.append(f"{recipient}: HTTP {resp.status}")
        except urllib.error.HTTPError as exc:
            try:
                err = json.loads(exc.read().decode()).get("message", "Unknown error")
            except Exception:
                err = str(exc)
            errors.append(f"{recipient}: {err}")
        except Exception as exc:
            errors.append(f"{recipient}: {exc}")

    if errors:
        st.warning("Some alerts could not be delivered:\n" + "\n".join(f"• {e}" for e in errors), icon="📧")

    return success_count == len(family_emails)


def build_daily_alert_details(data, pred, proba):
    status_map = {0: "Needs Attention", 1: "Plateauing", 2: "On Track"}
    bp_sys, bp_dia = data.get("bp_systolic", 0), data.get("bp_diastolic", 0)
    bp_note = " CRITICAL" if bp_sys >= 180 or bp_dia >= 120 else \
              " HIGH"     if bp_sys >= 140 or bp_dia >= 90  else \
              " LOW"      if bp_sys < 90  or bp_dia < 60   else ""
    return {
        "Recovery Status":       status_map.get(pred, "Unknown"),
        "Blood Pressure":        f"{bp_sys}/{bp_dia} mmHg{bp_note}",
        "Pain Level":            f"{data.get('pain','?')}/10",
        "Fatigue Level":         f"{data.get('fatigue','?')}/10",
        "Mobility Score":        f"{data.get('mobility','?')}/10",
        "Balance Score":         f"{data.get('balance','?')}/10",
        "Exercises Completed":   f"{data.get('exercises_done','?')}/10",
        "Sleep Quality":         f"{data.get('sleep_quality','?')}/10",
        "Mood":                  f"{data.get('mood','?')}/10",
        "Confidence (On Track)": f"{proba[2]:.0%}" if len(proba) > 2 else "--",
    }
