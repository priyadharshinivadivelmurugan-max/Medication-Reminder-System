import re
import json
from paddleocr import PaddleOCR
from rapidfuzz import process

# -----------------------------------
# Medicine Database
# Replace this with a CSV/database
# -----------------------------------

MEDICINE_DB = {
    "DOLO": "PARACETAMOL",
    "CROCIN": "PARACETAMOL",
    "CALPOL": "PARACETAMOL",
    "AZEE": "AZITHROMYCIN",
    "AUGMENTIN": "AMOXICILLIN + CLAVULANIC ACID",
    "PAN": "PANTOPRAZOLE",
    "ECOSPRIN": "ASPIRIN",
    "TELMA": "TELMISARTAN",
    "GLYCOMET": "METFORMIN",
    "MONTAIR": "MONTELUKAST"
}

MEDICINE_NAMES = list(MEDICINE_DB.keys())

# -----------------------------------
# Initialize PaddleOCR
# -----------------------------------

ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en"
)

# -----------------------------------
# Clean OCR text
# -----------------------------------

def clean_text(text):

    text = text.upper()

    # Common OCR mistakes
    text = text.replace("|", "I")
    text = text.replace("0", "O")

    text = re.sub(r"[^A-Z0-9.+ ]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# -----------------------------------
# Extract Strength
# -----------------------------------

def extract_strength(text):

    patterns = [
        r"\d+(?:\.\d+)?\s?MG",
        r"\d+(?:\.\d+)?\s?ML",
        r"\d+(?:\.\d+)?\s?MCG",
        r"\d+(?:\.\d+)?\s?G",
        r"\d+(?:\.\d+)?\s?IU"
    ]

    for pattern in patterns:
        match = re.search(pattern, text)

        if match:
            return match.group().replace(" ", "")

    # Handle blister text like DOLO 650
    match = re.search(r"[A-Z]+\s(\d{2,4})", text)

    if match:
        return match.group(1) + "MG"

    return "unknown"

# -----------------------------------
# Find Brand Name
# -----------------------------------

def find_brand(text):

    tokens = text.split()

    best_brand = "unknown"
    best_score = 0

    for token in tokens:

        match = process.extractOne(
            token,
            MEDICINE_NAMES
        )

        if match is None:
            continue

        brand, score, _ = match

        if score > best_score:
            best_score = score
            best_brand = brand

    if best_score >= 85:
        return best_brand, round(best_score / 100, 2)

    return "unknown", 0.0

# -----------------------------------
# Main Function
# -----------------------------------

def interpret_blister(image_path):

    result = ocr.ocr(image_path)

    text_list = []

    for page in result:

        if page is None:
            continue

        for line in page:
            text_list.append(line[1][0])

    raw_text = " ".join(text_list)

    cleaned = clean_text(raw_text)

    brand, confidence = find_brand(cleaned)

    generic = MEDICINE_DB.get(
        brand,
        "unknown"
    )

    strength = extract_strength(cleaned)

    return {
        "brand_name": brand,
        "generic_name": generic,
        "strength": strength,
        "confidence": confidence,
        "source_text": cleaned
    }

# -----------------------------------
# Example
# -----------------------------------

if __name__ == "__main__":

    image_path = "blister.jpg"

    result = interpret_blister(image_path)

    print(json.dumps(result, indent=4))
