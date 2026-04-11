"""
alerts.py — Stroke Recovery Monitor v2.0
=========================================
Sends email alerts to family members when:
  - Blood pressure is above/below normal range
  - Patient completes their daily check-in
  - Recovery status is "Needs Attention"
  - Patient misses a daily log (can be scheduled)

Uses Python's built-in smtplib — no API key needed.
Family must configure their Gmail in Streamlit secrets.

Author: Samuel Oluwakoya
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import streamlit as st


def _get_smtp_config():
    """
    Read email credentials from Streamlit secrets.
    In Streamlit Cloud: Settings → Secrets → add:
        SENDER_EMAIL = "your-gmail@gmail.com"
        SENDER_PASSWORD = "your-app-password"
    Locally: create .streamlit/secrets.toml with same keys.
    """
    try:
        email    = st.secrets["SENDER_EMAIL"]
        password = st.secrets["SENDER_PASSWORD"]
        return email, password
    except Exception:
        return None, None


def send_alert_email(patient_name: str, family_emails: list,
                     alert_type: str, message: str,
                     details: dict = None) -> bool:
    """
    Sends a formatted HTML email alert to all family members.
    Returns True if sent, False if email not configured.
    """
    sender_email, sender_password = _get_smtp_config()
    if not sender_email or not family_emails:
        return False

    # Choose subject and colour by alert type
    config = {
        "bp_critical": ("🚨 URGENT — Blood Pressure Emergency", "#dc2626"),
        "bp_high":     ("⚠️ Alert — High Blood Pressure Detected", "#d97706"),
        "bp_low":      ("⚠️ Alert — Low Blood Pressure Detected", "#2563eb"),
        "needs_attention": ("🔴 Recovery Alert — Needs Attention", "#dc2626"),
        "daily_complete":  ("✅ Daily Check-In Complete", "#059669"),
        "exercise_done":   ("🏃 Exercise Session Completed", "#4f46e5"),
        "plateau":         ("🟡 Recovery Update — Plateauing", "#d97706"),
        "on_track":        ("🟢 Recovery Update — On Track", "#059669"),
    }
    subject, accent = config.get(alert_type, ("📊 Recovery Update", "#4f46e5"))

    # Build details table
    details_html = ""
    if details:
        rows = "".join(
            f"<tr><td style='padding:6px 12px;color:#6b7280;font-size:14px;'>{k}</td>"
            f"<td style='padding:6px 12px;font-weight:600;font-size:14px;'>{v}</td></tr>"
            for k, v in details.items()
        )
        details_html = f"""
        <table style='width:100%;border-collapse:collapse;margin-top:16px;
                      border:1px solid #e5e7eb;border-radius:8px;overflow:hidden;'>
            <tbody>{rows}</tbody>
        </table>"""

    timestamp = datetime.now().strftime("%A, %d %B %Y at %I:%M %p")

    html = f"""
    <!DOCTYPE html>
    <html>
    <body style='margin:0;padding:0;background:#f9fafb;font-family:Arial,sans-serif;'>
      <div style='max-width:560px;margin:30px auto;background:white;
                  border-radius:16px;overflow:hidden;
                  box-shadow:0 2px 12px rgba(0,0,0,0.08);'>

        <!-- Header -->
        <div style='background:{accent};padding:28px 32px;'>
          <div style='font-size:22px;font-weight:700;color:white;'>{subject}</div>
          <div style='font-size:14px;color:rgba(255,255,255,0.85);margin-top:6px;'>
            Stroke Recovery Monitor · {timestamp}
          </div>
        </div>

        <!-- Body -->
        <div style='padding:28px 32px;'>
          <p style='font-size:16px;color:#374151;line-height:1.6;margin-top:0;'>
            This is an automated update regarding <strong>{patient_name}</strong>'s
            stroke recovery monitoring.
          </p>

          <div style='background:#f9fafb;border-left:4px solid {accent};
                      border-radius:0 8px 8px 0;padding:16px 20px;margin:20px 0;'>
            <div style='font-size:15px;color:#1f2937;line-height:1.6;'>{message}</div>
          </div>

          {details_html}

          <p style='font-size:13px;color:#9ca3af;margin-top:24px;line-height:1.6;'>
            View the full dashboard by logging in at the Stroke Recovery Monitor app
            with <strong>{patient_name}</strong>'s username and your family access code.
          </p>
        </div>

        <!-- Footer -->
        <div style='background:#f9fafb;padding:18px 32px;
                    border-top:1px solid #e5e7eb;'>
          <div style='font-size:12px;color:#9ca3af;'>
            Built by <strong>Samuel Oluwakoya</strong> · Stroke Recovery Monitor v2.0<br>
            This is an automated alert. Do not reply to this email.
          </div>
        </div>

      </div>
    </body>
    </html>
    """

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = sender_email
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            for recipient in family_emails:
                msg["To"] = recipient
                server.sendmail(sender_email, recipient, msg.as_string())
        return True
    except Exception as e:
        # Fail silently — don't crash the app if email fails
        return False


def build_daily_alert_details(data: dict, pred: int, proba: list) -> dict:
    """Build the details dict for the daily completion email."""
    status_map = {0: "🔴 Needs Attention", 1: "🟡 Plateauing", 2: "🟢 On Track"}
    bp_note = ""
    if data.get("bp_systolic", 0) >= 140:
        bp_note = " ⚠️ HIGH"
    elif data.get("bp_systolic", 0) < 90:
        bp_note = " ⚠️ LOW"

    return {
        "Recovery Status":    status_map.get(pred, "Unknown"),
        "Blood Pressure":     f"{data.get('bp_systolic','?')}/{data.get('bp_diastolic','?')} mmHg{bp_note}",
        "Pain Level":         f"{data.get('pain', '?')}/10",
        "Fatigue Level":      f"{data.get('fatigue', '?')}/10",
        "Mobility Score":     f"{data.get('mobility', '?')}/10",
        "Balance Score":      f"{data.get('balance', '?')}/10",
        "Exercises Completed":f"{data.get('exercises_done', '?')}/10",
        "Sleep Quality":      f"{data.get('sleep_quality', '?')}/10",
        "Mood":               f"{data.get('mood', '?')}/10",
        "Confidence (On Track)": f"{proba[2]:.0%}" if len(proba) > 2 else "—",
    }
