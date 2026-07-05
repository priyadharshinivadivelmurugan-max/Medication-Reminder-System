import re

TIME_SLOTS = {
    "morning": "08:00",
    "afternoon": "13:00",
    "evening": "18:00",
    "night": "20:00"
}

def parse_frequency(freq):
    freq = freq.lower().strip()

    if "stat" in freq:
        return ["stat"]


    # Common abbreviations
    if freq == "od":
        return ["morning"]
    if freq == "bd":
        return ["morning", "night"]
    if freq == "tds":
        return ["morning", "afternoon", "night"]

    # Numeric patterns (1-0-0-1 etc.)
    parts = re.split(r'[-\s]', freq)
    times = ["morning", "afternoon", "evening", "night"]

    schedule = []
    for p, t in zip(parts, times):
        try:
            if float(p) > 0:
                schedule.append(t)
        except:
            pass

    return schedule

# def apply_food_relation(time, relation):
#     relation = relation.lower()

#     if "after" in relation:
#         return time + " (after food)"
#     if "before" in relation:
#         return time + " (before food)"
#     return time
from datetime import datetime, timedelta

def apply_food_relation(time, relation):
    relation = relation.lower()

    base_time = datetime.strptime(time, "%H:%M")

    if "before" in relation:
        reminder_time = base_time - timedelta(minutes=30)
        return reminder_time.strftime("%H:%M") + " (before food)"

    if "after" in relation:
        return time + " (after food)"

    return time

def parse_duration(duration_text, default_days=1):
    if not duration_text:
        return default_days

    text = duration_text.lower()

    # Only consider numbers linked to days
    if "day" in text:
        nums = re.findall(r'\d+', text)
        if nums:
            return int(nums[0])

    return default_days



def build_schedule(medicine):
    times = []

    # Injection handling (independent of frequency)
    if medicine.get("dosage_form") == "injection":
        return {
        "medicine": medicine.get("generic_name") or medicine.get("medicine_name"),
        "brand": medicine.get("brand_name", medicine.get("medicine_name", "unknown")),
        "strength": medicine.get("strength", "as prescribed"),
        "times": ["Administer as directed (Injection)"],
        "duration_days": 1
    }


    slots = parse_frequency(medicine["frequency"])

# STAT injection handling
    if "stat" in slots:
        return {
            "medicine": medicine.get("generic_name") or medicine.get("medicine_name"),
            "brand": medicine.get("brand_name", medicine.get("medicine_name", "unknown")),
            "strength": medicine.get("strength", "as prescribed"),
            "times": ["Administer as directed (Injection)"],
            "duration_days": 1
        }


    for slot in slots:
        base_time = TIME_SLOTS[slot]
        final_time = apply_food_relation(
            base_time,
            medicine.get("food_relation", "")
        )
        times.append(final_time)

    duration = parse_duration(medicine.get("duration"))

    return {
        "medicine": medicine.get("generic_name") or medicine.get("medicine_name"),
        "brand": medicine.get("brand_name", medicine.get("medicine_name", "unknown")),
        "strength": medicine.get("strength", "as prescribed"),
        "times": times,
        "duration_days": duration
    }

def generate_schedule(validated_medicines):
    schedule = []
    for med in validated_medicines:
        schedule.append(build_schedule(med))
    return schedule
