import os
import smtplib
import ssl
from email.mime.text import MIMEText
from email.utils import formataddr
from email.mime.multipart import MIMEMultipart  # New line
from email.mime.base import MIMEBase  # New line
from email import encoders  # New line

try:
        from dotenv import load_dotenv
except ImportError:  # pragma: no cover - lightweight fallback if package missing
        load_dotenv = None

if load_dotenv:
        load_dotenv()

def _get_env(name, default=None, *, required=False):
        value = os.getenv(name, default)
        if required and not value:
                raise RuntimeError(f"Environment variable '{name}' is required but not set.")
        return value

def _get_env_list(name):
        raw = _get_env(name, "")
        return [item.strip() for item in raw.split(",") if item.strip()]

# User configuration loaded from environment / .env file
sender_email = _get_env('CRASH_MAIL_USER', required=True)
sender_name = _get_env('CRASH_MAIL_SENDER_NAME', sender_email)
password = _get_env('CRASH_MAIL_PASS', required=True)
receiver_emails = _get_env_list('CRASH_MAIL_TO')
receiver_names = _get_env_list('CRASH_MAIL_TO_NAMES') or receiver_emails
server = None

# Email body
#email_html = open('email.html')
#email_body = email_html.read()

filename = 'output/crash.jpg'
def sendalert():
        if not receiver_emails:
                raise RuntimeError("No receivers configured. Set CRASH_MAIL_TO in your environment.")
        if len(receiver_emails) != len(receiver_names):
                raise RuntimeError("CRASH_MAIL_TO and CRASH_MAIL_TO_NAMES must have the same number of entries.")
        for receiver_email, receiver_name in zip(receiver_emails, receiver_names):
                print("Sending email...")
                # Configurating user's info
                msg = MIMEMultipart()
                msg['To'] = formataddr((receiver_name, receiver_email))
                msg['From'] = formataddr((sender_name, sender_email))
                msg['Subject'] = 'Crash detected at the location xyz'

                #msg.attach(MIMEText(email_body, 'html'))

                try:
                    # Open PDF file in binary mode
                    with open(filename, "rb") as attachment:
                                    part = MIMEBase("application", "octet-stream")
                                    part.set_payload(attachment.read())

                    # Encode file in ASCII characters to send by email
                    encoders.encode_base64(part)

                    # Add header as key/value pair to attachment part
                    part.add_header(
                            "Content-Disposition",
                            f"attachment; filename= {filename}",
                    )

                    msg.attach(part)
                except Exception as e:
                        print(f'Oh no! We didnt found the attachment!\n{e}')
                        break

                email_sent = False
                try:
                        # Creating a SMTP session | use 587 with TLS, 465 SSL and 25
                        server = smtplib.SMTP('smtp.gmail.com', 587)
                        # Encrypts the email
                        context = ssl.create_default_context()
                        server.starttls(context=context)
                        # We log in into our Google account
                        server.login(sender_email, password)
                        # Sending email from sender, to receiver with the email body
                        server.sendmail(sender_email, receiver_email, msg.as_string())
                        email_sent = True
                        print('Email sent!')
                except Exception as e:
                        print(f'\n EMAIL ERROR: Failed to send email\n   Error: {e}\n')
                        break
                finally:
                        print('Closing the server...')
                        if server:
                                server.quit()
                                server = None
                        if not email_sent:
                                print('Email was NOT sent due to error above.')

