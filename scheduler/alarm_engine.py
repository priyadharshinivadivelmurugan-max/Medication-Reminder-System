import time
import json
import os
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, messaging

SCHEDULE_FILE = "output/schedule.json"
DEVICE_TOKEN = "dbY7NSRiQKubJUe1iJ2Re_:APA91bGeyhpg2EYFyHWQmgHyx4dzl_p_Z693uFgjCl6eXRO9N--SKPIgL6vQtUqzCMXFCLSMqSMvNj1BQ9ptQ_NUetwZZdRvuRz2n7Ej4FRmYOsoppHT4OQ"

cred = credentials.Certificate("prescriptionvoice-firebase-adminsdk-fbsvc-28326deb6d.json")
firebase_admin.initialize_app(cred)

def extract_time(raw_time):
    parts = raw_time.split()
    if parts and ":" in parts[0]:
        return parts[0]
    return None

def send_push(medicine_names, language):

    if language == "Tamil":
        body = f"தயவுசெய்து {medicine_names} மருந்துகளை எடுத்துக்கொள்ளுங்கள்"
    else:
        body = f"Please take {medicine_names} medicines now"

    message = messaging.Message(
        data={
            "title": "Medicine Reminder",
            "body": body
        },
        token=DEVICE_TOKEN
    )

    messaging.send(message)
    print("Push sent:", body)

def run_alarm_service():
    print("Alarm service started...")
    triggered_today = set()

    while True:
        if not os.path.exists(SCHEDULE_FILE):
            time.sleep(10)
            continue

        with open(SCHEDULE_FILE, "r") as f:
            #schedule = json.load(f)
            data = json.load(f)
            schedule = data.get("schedule", [])
            language = data.get("language", "English")

        now = datetime.now()
        current_time = now.strftime("%H:%M")
        today = now.strftime("%Y-%m-%d")

        medicines_to_take = []

        for med in schedule:
            for raw_time in med.get("times", []):
                alarm_time = extract_time(raw_time)
                if not alarm_time:
                    continue

                key = f"{today}_{alarm_time}"

                if current_time == alarm_time and key not in triggered_today:
                    medicines_to_take.append(med["medicine"])

        if medicines_to_take:
            meds = ", ".join(medicines_to_take)
            send_push(meds, language)
            triggered_today.add(f"{today}_{current_time}")

        time.sleep(60)

if __name__ == "__main__":
    run_alarm_service()

