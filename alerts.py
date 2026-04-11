"""
alerts.py — Stroke Recovery Monitor v2.0
=========================================
Email alerts via Resend (https://resend.com).

WHY RESEND INSTEAD OF GMAIL SMTP:
  - No Gmail app passwords, no 2FA setup, no SMTP config
  - Patients and family just type their email addresses — nothing else
  - You (Samuel) create one free Resend account, get one API key,
    paste it into Streamlit Secrets once. Done for all users forever.
  - Free tier: 3,000 emails/month, 100/day

SETUP (one time, done by Samuel):
  1. Go to resend.com → Sign up free
  2. Settings → API Keys → Create API Key → copy it
  3. In Streamlit Cloud: App Settings → Secrets → add:
        RESEND_API_KEY = "re_xxxxxxxxxxxxxxxxxxxx"
  4. (Optional but recommended) Verify a sender domain or use
     the default onboarding@resend.dev for testing

That is it. Patients never need to do anything technical.

Author: Samuel Oluwakoya
"""

import streamlit as st
from datetime import datetime

# Lazy import — only loaded if resend is installed
# Falls back gracefully if not installed (shows clear error)
def _get_resend():
    try:
        import resend
        return resend
    except ImportError:
        return None


def _get_api_key() -> str | None:
    """Read Resend API key from Streamlit secrets. Returns None if not set."""
    try:
        return st.secrets.get("RESEND_API_KEY", None)
    except Exception:
        return None


def _get_sender_email() -> str:
    """
    The FROM address. Use your own verified domain email for production.
    For testing, Resend provides onboarding@resend.dev automatically.
    You can also set SENDER_EMAIL in secrets to override.
    """
    try:
        return st.secrets.get("SENDER_EMAIL", "Stroke Recovery Monitor <onboarding@resend.dev>")
    except Exception:
        return "Stroke Recovery Monitor <onboarding@resend.dev>"


def send_alert_email(
    patient_name: str,
    family_emails: list,
    alert_type: str,
    message: str,
    details: dict = None,
) -> bool:
    """
    Send a formatted HTML alert email to all family addresses.
    Returns True on success, False on failure.
    Shows a visible UI error so the user knows if something went wrong.
    """
    resend = _get_resend()
    api_key = _get_api_key()

    # ── Pre-flight checks ─────────────────────────────────────────────────
    if resend is None:
        st.error(
            "📦 The `resend` package is not installed. "
            "Add `resend` to your requirements.txt and redeploy."
        )
        return False

    if not api_key:
        st.warning(
            "⚠️ Email alerts are not configured yet. "
            "Add your Resend API key to Streamlit Secrets to enable alerts.",
            icon="📧",
        )
        return False

    if not family_emails:
        return False

    resend.api_key = api_key

    # ── Email subject and accent colour ───────────────────────────────────
    config = {
        "bp_critical":     ("🚨 URGENT — Blood Pressure Emergency",       "#dc2626"),
        "bp_high":         ("⚠️ Alert — High Blood Pressure Detected",    "#d97706"),
        "bp_low":          ("⚠️ Alert — Low Blood Pressure",              "#2563eb"),
        "needs_attention": ("🔴 Recovery Alert — Needs Attention Today",  "#dc2626"),
        "daily_complete":  ("✅ Daily Check-In Completed",                 "#059669"),
        "plateau":         ("🟡 Recovery Update — Plateau Phase",         "#d97706"),
        "on_track":        ("🟢 Recovery Update — On Track",              "#059669"),
    }
    subject, accent = config.get(alert_type, ("📊 Recovery Update", "#4f46e5"))
    timestamp = datetime.now().strftime("%A, %d %B %Y at %I:%M %p")

    # ── Detail table ──────────────────────────────────────────────────────
    details_html = ""
    if details:
        rows = "".join(
            f"<tr>"
            f"<td style='padding:8px 14px;color:#6b7280;font-size:14px;"
            f"border-bottom:1px solid #f3f4f6;white-space:nowrap'>{k}</td>"
            f"<td style='padding:8px 14px;font-weight:600;font-size:14px;"
            f"border-bottom:1px solid #f3f4f6'>{v}</td>"
            f"</tr>"
            for k, v in details.items()
        )
        details_html = f"""
        <table style='width:100%;border-collapse:collapse;margin-top:16px;
                      border:1px solid #e5e7eb;border-radius:8px;overflow:hidden;'>
          <tbody>{rows}</tbody>
        </table>"""

    html = f"""<!DOCTYPE html>
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

    # ── Send to each recipient ─────────────────────────────────────────────
    sender = _get_sender_email()
    errors = []
    success_count = 0

    for recipient in family_emails:
        try:
            resend.Emails.send({
                "from":    sender,
                "to":      [recipient],
                "subject": subject.replace("🚨","").replace("⚠️","").replace("✅","")
                           .replace("🔴","").replace("🟡","").replace("🟢","").strip(),
                "html":    html,
            })
            success_count += 1
        except Exception as exc:
            errors.append(f"{recipient}: {type(exc).__name__}: {str(exc)[:120]}")

    if errors:
        st.warning(
            "⚠️ Some alerts could not be delivered:\n"
            + "\n".join(f"• {e}" for e in errors),
            icon="📧",
        )

    return success_count == len(family_emails)


def build_daily_alert_details(data: dict, pred: int, proba: list) -> dict:
    """Build the detail table for the daily completion email."""
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
        "Recovery Status":       status_map.get(pred, "Unknown"),
        "Blood Pressure":        f"{bp_sys}/{bp_dia} mmHg{bp_note}",
        "Pain Level":            f"{data.get('pain','?')}/10",
        "Fatigue Level":         f"{data.get('fatigue','?')}/10",
        "Mobility Score":        f"{data.get('mobility','?')}/10",
        "Balance Score":         f"{data.get('balance','?')}/10",
        "Exercises Completed":   f"{data.get('exercises_done','?')}/10",
        "Sleep Quality":         f"{data.get('sleep_quality','?')}/10",
        "Mood":                  f"{data.get('mood','?')}/10",
        "Confidence (On Track)": f"{proba[2]:.0%}" if len(proba) > 2 else "—",
    }
