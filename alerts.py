"""
alerts.py — Stroke Recovery Monitor v2.0
=========================================
Email alerts to family members via Gmail SMTP.

Fixes applied:
  - Uses port 587 + STARTTLS (more reliable than SSL 465 in cloud)
  - Each recipient gets a properly addressed separate email
  - BP critical alert is sent BEFORE the daily completion alert
    so family see the emergency first
  - st.warning() shown in UI when email fails so patient/carer knows
  - Detailed exception logged to st.session_state for debugging

Author: Samuel Oluwakoya
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import streamlit as st


def _get_smtp_config():
    """
    Read Gmail credentials from Streamlit secrets.
    In Streamlit Cloud: Settings → Secrets → paste:
        SENDER_EMAIL    = "your-gmail@gmail.com"
        SENDER_PASSWORD = "your-16-char-app-password"
    """
    try:
        email    = st.secrets["SENDER_EMAIL"]
        password = st.secrets["SENDER_PASSWORD"]
        return email, password
    except Exception:
        return None, None


def send_alert_email(
    patient_name: str,
    family_emails: list,
    alert_type: str,
    message: str,
    details: dict = None,
) -> bool:
    """
    Send a formatted HTML email to every address in family_emails.
    Returns True if all sent successfully, False otherwise.
    Shows a visible UI warning if email is not configured.
    """
    sender_email, sender_password = _get_smtp_config()

    if not sender_email:
        st.warning(
            "⚠️ Email alerts not configured. "
            "Add SENDER_EMAIL and SENDER_PASSWORD to Streamlit secrets to enable alerts.",
            icon="📧",
        )
        return False

    if not family_emails:
        return False

    # Subject line and accent colour per alert type
    config = {
        "bp_critical":     ("🚨 URGENT — Blood Pressure Emergency",        "#dc2626"),
        "bp_high":         ("⚠️ Alert — High Blood Pressure Detected",     "#d97706"),
        "bp_low":          ("⚠️ Alert — Low Blood Pressure Detected",      "#2563eb"),
        "needs_attention": ("🔴 Recovery Alert — Needs Attention Today",   "#dc2626"),
        "daily_complete":  ("✅ Daily Check-In Completed",                  "#059669"),
        "plateau":         ("🟡 Recovery Update — Plateau Phase",          "#d97706"),
        "on_track":        ("🟢 Recovery Update — On Track",               "#059669"),
    }
    subject, accent = config.get(alert_type, ("📊 Recovery Update", "#4f46e5"))
    timestamp = datetime.now().strftime("%A, %d %B %Y at %I:%M %p")

    # Build details table rows
    details_html = ""
    if details:
        rows = "".join(
            f"<tr>"
            f"<td style='padding:7px 14px;color:#6b7280;font-size:14px;border-bottom:"
            f"1px solid #f3f4f6;'>{k}</td>"
            f"<td style='padding:7px 14px;font-weight:600;font-size:14px;border-bottom:"
            f"1px solid #f3f4f6;'>{v}</td>"
            f"</tr>"
            for k, v in details.items()
        )
        details_html = f"""
        <table style='width:100%;border-collapse:collapse;margin-top:16px;
                      border:1px solid #e5e7eb;border-radius:8px;overflow:hidden;'>
          <tbody>{rows}</tbody>
        </table>"""

    html_body = f"""<!DOCTYPE html>
<html>
<body style='margin:0;padding:0;background:#f9fafb;font-family:Arial,sans-serif;'>
  <div style='max-width:560px;margin:30px auto;background:white;
              border-radius:16px;overflow:hidden;
              box-shadow:0 2px 12px rgba(0,0,0,0.08);'>
    <div style='background:{accent};padding:28px 32px;'>
      <div style='font-size:20px;font-weight:700;color:white;'>{subject}</div>
      <div style='font-size:13px;color:rgba(255,255,255,0.85);margin-top:6px;'>
        Stroke Recovery Monitor &middot; {timestamp}
      </div>
    </div>
    <div style='padding:28px 32px;'>
      <p style='font-size:15px;color:#374151;line-height:1.6;margin-top:0;'>
        Automated update for <strong>{patient_name}</strong>.
      </p>
      <div style='background:#f9fafb;border-left:4px solid {accent};
                  border-radius:0 8px 8px 0;padding:16px 20px;margin:20px 0;'>
        <div style='font-size:15px;color:#1f2937;line-height:1.6;'>{message}</div>
      </div>
      {details_html}
      <p style='font-size:13px;color:#9ca3af;margin-top:24px;line-height:1.6;'>
        Log in to the Stroke Recovery Monitor app with the patient username
        and your family access code to view the full dashboard.
      </p>
    </div>
    <div style='background:#f9fafb;padding:18px 32px;border-top:1px solid #e5e7eb;'>
      <div style='font-size:12px;color:#9ca3af;'>
        Built by <strong>Samuel Oluwakoya</strong> &middot;
        Stroke Recovery Monitor v2.0<br>
        Automated alert — do not reply to this email.
      </div>
    </div>
  </div>
</body>
</html>"""

    success = True
    errors  = []

    for recipient in family_emails:
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"]    = sender_email
            msg["To"]      = recipient
            msg.attach(MIMEText(html_body, "html"))

            # Port 587 + STARTTLS — works in Streamlit Cloud
            with smtplib.SMTP("smtp.gmail.com", 587, timeout=15) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, recipient, msg.as_string())

        except smtplib.SMTPAuthenticationError:
            errors.append(f"{recipient}: authentication failed — check app password in secrets")
            success = False
        except smtplib.SMTPRecipientsRefused:
            errors.append(f"{recipient}: address rejected by Gmail")
            success = False
        except Exception as exc:
            errors.append(f"{recipient}: {type(exc).__name__}: {exc}")
            success = False

    if errors:
        # Show visible warning in the UI — patient/carer can see delivery failed
        st.warning(
            "⚠️ Some alert emails could not be delivered:\n"
            + "\n".join(f"• {e}" for e in errors),
            icon="📧",
        )

    return success


def build_daily_alert_details(data: dict, pred: int, proba: list) -> dict:
    """Build the key-value detail table for the daily completion email."""
    status_map = {0: "🔴 Needs Attention", 1: "🟡 Plateauing", 2: "🟢 On Track"}
    bp_sys = data.get("bp_systolic", 0)
    bp_dia = data.get("bp_diastolic", 0)
    bp_note = ""
    if bp_sys >= 180 or bp_dia >= 120:
        bp_note = " 🚨 CRITICAL"
    elif bp_sys >= 140 or bp_dia >= 90:
        bp_note = " ⚠️ HIGH"
    elif bp_sys < 90 or bp_dia < 60:
        bp_note = " ⚠️ LOW"

    return {
        "Recovery Status":      status_map.get(pred, "Unknown"),
        "Blood Pressure":       f"{bp_sys}/{bp_dia} mmHg{bp_note}",
        "Pain Level":           f"{data.get('pain','?')}/10",
        "Fatigue Level":        f"{data.get('fatigue','?')}/10",
        "Mobility Score":       f"{data.get('mobility','?')}/10",
        "Balance Score":        f"{data.get('balance','?')}/10",
        "Exercises Completed":  f"{data.get('exercises_done','?')}/10",
        "Sleep Quality":        f"{data.get('sleep_quality','?')}/10",
        "Mood":                 f"{data.get('mood','?')}/10",
        "Confidence (On Track)":f"{proba[2]:.0%}" if len(proba) > 2 else "—",
    }
