import re
import json
from paddleocr import PaddleOCR
from rapidfuzz import process

# -----------------------------
# Medicine Database
# -----------------------------
MEDICINE_DB = {
    "DOLO": "tablet",
    "CROCIN": "tablet",
    "PAN": "tablet",
    "AZEE": "tablet",
    "AUGMENTIN": "tablet",
    "ECOSPRIN": "tablet",
    "CALPOL": "tablet",
    "MONTAIR": "tablet",
    "TELMA": "tablet",
    "GLYCOMET": "tablet",
    "METFORMIN": "tablet",
    "PARACETAMOL": "tablet",
    "AMOXICILLIN": "capsule",
    "AZITHROMYCIN": "tablet",
    "PANTOPRAZOLE": "tablet",
    "CETRIZINE": "tablet"
}

MEDICINE_NAMES = list(MEDICINE_DB.keys())

IGNORE_WORDS = [
    "BP", "PULSE", "TEMP", "SPO2", "WEIGHT",
    "CBC", "XRAY", "X-RAY", "MRI", "CT",
    "URINE", "BLOOD", "DIAGNOSIS", "INVESTIGATION",
    "FOLLOW", "REVIEW", "DATE", "AGE", "SEX"
]

ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en"
)


def clean_text(text):
    text = text.upper()
    text = text.replace("|", "I")
    text = text.replace("(", " ")
    text = text.replace(")", " ")

    text = re.sub(r"[^A-Z0-9./+\- ]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def medicine_match(line):
    tokens = line.split()

    best_name = "unknown"
    best_score = 0

    for token in tokens:
        match = process.extractOne(token, MEDICINE_NAMES)

        if match is None:
            continue

        name, score, _ = match

        if score > best_score:
            best_score = score
            best_name = name

    if best_score >= 85:
        return best_name, best_score / 100

    return "unknown", 0.0


def extract_frequency(line):

    patterns = [
        r"\b\d-\d-\d\b",
        r"\bOD\b",
        r"\bBD\b",
        r"\bTDS\b",
        r"\bQID\b",
        r"\bSOS\b",
        r"\bHS\b"
    ]

    for pattern in patterns:
        match = re.search(pattern, line)
        if match:
            return match.group()

    return "unknown"


def extract_duration(line):

    pattern = r"(\d+\s*(?:DAY|DAYS|WEEK|WEEKS|MONTH|MONTHS))"

    match = re.search(pattern, line)

    if match:
        return match.group(1)

    return "unknown"


def extract_food_relation(line):

    if re.search(r"\bAF\b", line):
        return "AF"

    if re.search(r"\bBF\b", line):
        return "BF"

    return "unknown"


def extract_dosage_form(line, medicine):

    line = line.upper()

    if "TAB" in line or "TABLET" in line:
        return "tablet"

    if "CAP" in line or "CAPSULE" in line:
        return "capsule"

    if "SYRUP" in line or "SYP" in line:
        return "syrup"

    if "OINT" in line:
        return "ointment"

    if "SOLN" in line or "SOLUTION" in line:
        return "solution"

    if "INJ" in line or "INJECTION" in line:
        return "injection"

    return MEDICINE_DB.get(medicine, "unknown")


def calculate_confidence(
    medicine,
    frequency,
    duration,
    food_relation
):

    score = 0.0

    if medicine != "unknown":
        score += 0.4

    if frequency != "unknown":
        score += 0.2

    if duration != "unknown":
        score += 0.2

    if food_relation != "unknown":
        score += 0.2

    return round(score, 2)


def interpret_prescription(image_path):

    result = ocr.ocr(image_path)

    lines = []

    for page in result:

        if page is None:
            continue

        for item in page:
            lines.append(item[1][0])

    schedules = []

    for raw_line in lines:

        line = clean_text(raw_line)

        if len(line) < 3:
            continue

        if any(word in line for word in IGNORE_WORDS):
            continue

        medicine, _ = medicine_match(line)

        if medicine == "unknown":
            continue

        frequency = extract_frequency(line)

        duration = extract_duration(line)

        food_relation = extract_food_relation(line)

        dosage_form = extract_dosage_form(
            line,
            medicine
        )

        confidence = calculate_confidence(
            medicine,
            frequency,
            duration,
            food_relation
        )

        schedules.append({
            "medicine_name": medicine,
            "dosage_form": dosage_form,
            "frequency": frequency,
            "duration": duration,
            "food_relation": food_relation,
            "confidence": confidence,
            "source_text": raw_line
        })

    return {
        "schedules": schedules,
        "needs_manual_review": len(schedules) == 0
    }


if __name__ == "__main__":

    image_path = "prescription.jpg"

    result = interpret_prescription(image_path)

    print(json.dumps(result, indent=4))
