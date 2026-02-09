import smtplib
import resend
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from database import get_smtp_settings

def send_notification_email(subject, data, type='contact'):
    """Send a notification email using Resend (HTTPS) or SMTP fallback"""
    try:
        settings = get_smtp_settings()
        
        resend_api_key = settings.get('resend_api_key')
        sender_email = settings.get('smtp_email') # Still used for 'From'
        receiver_email = settings.get('receiver_email') or sender_email
        smtp_password = settings.get('smtp_password')

        # Format body
        if type == 'contact':
            body_text = f"New Contact Submission:\nName: {data.get('name')}\nEmail: {data.get('email')}\nPhone: {data.get('phone')}\nSubject: {data.get('subject')}\n\nMessage:\n{data.get('message')}"
            body_html = f"<h3>New Contact Submission</h3><p><b>Name:</b> {data.get('name')}</p><p><b>Email:</b> {data.get('email')}</p><p><b>Phone:</b> {data.get('phone')}</p><p><b>Subject:</b> {data.get('subject')}</p><p><b>Message:</b><br>{data.get('message')}</p>"
        else:
            body_text = f"New Enrollment Submission:\nName: {data.get('name')}\nEmail: {data.get('email')}\nPhone: {data.get('phone')}\nCourse: {data.get('course')}\n\nMessage:\n{data.get('message', 'N/A')}"
            body_html = f"<h3>New Enrollment Submission</h3><p><b>Name:</b> {data.get('name')}</p><p><b>Email:</b> {data.get('email')}</p><p><b>Phone:</b> {data.get('phone')}</p><p><b>Course:</b> {data.get('course')}</p><p><b>Message:</b><br>{data.get('message', 'N/A')}</p>"

        # METHOD 1: RESEND (Best for Railway)
        if resend_api_key:
            try:
                print("📧 Attempting to send via Resend API...")
                resend.api_key = resend_api_key
                
                params = {
                    "from": "Oriana Academy <onboarding@resend.dev>",
                    "to": [receiver_email],
                    "subject": f"Oriana Academy: {subject}",
                    "html": body_html,
                }
                
                # If they have a custom sender email verified in Resend, use it
                if sender_email and "@" in sender_email and "gmail.com" not in sender_email:
                    params["from"] = sender_email

                resend.Emails.send(params)
                print(f"✅ Email notification sent via Resend to {receiver_email}")
                return True, "Email sent via Resend"
            except Exception as re_err:
                print(f"⚠️ Resend failed: {re_err}. Trying SMTP fallback...")

        # METHOD 2: SMTP FALLBACK
        if all([sender_email, smtp_password, receiver_email]):
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = receiver_email
            msg['Subject'] = f"Oriana Academy: {subject}"
            msg.attach(MIMEText(body_text, 'plain'))

            try:
                print(f"📧 Attempting SMTP fallback (Port 587)...")
                server = smtplib.SMTP('smtp.gmail.com', 587, timeout=10)
                server.starttls()
                server.login(sender_email, smtp_password)
                server.send_message(msg)
                server.quit()
                print(f"✅ Email notification sent via SMTP to {receiver_email}")
                return True, "Email sent via SMTP"
            except Exception as e587:
                try:
                    print(f"⚠️ SMTP 587 failed, trying Port 465...")
                    server = smtplib.SMTP_SSL('smtp.gmail.com', 465, timeout=10)
                    server.login(sender_email, smtp_password)
                    server.send_message(msg)
                    server.quit()
                    print(f"✅ Email notification sent via SMTP SSL to {receiver_email}")
                    return True, "Email sent via SMTP SSL"
                except Exception as e465:
                    return False, f"Both Resend and SMTP failed. SMTP Error: {e465}"
        
        return False, "Neither Resend API nor SMTP credentials configured."

    except Exception as e:
        return False, str(e)
