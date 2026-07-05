import pandas as pd
import re
from rapidfuzz import fuzz


def normalize(text):
    if pd.isna(text):
        return ""
    return re.sub(r'[^a-z0-9]', '', str(text).lower())


def extract_strength(text):
    """
    Matches:
    650MG
    650 MG
    0.5MG
    5ML
    250MCG
    """
    pattern = r'(\d+(?:\.\d+)?)\s*(MG|ML|MCG|G|IU)'
    matches = re.findall(pattern, text.upper())

    strengths = set()

    for value, unit in matches:
        strengths.add(f"{value}{unit}")

    return strengths


def load_database(db_path):

    db = pd.read_excel(db_path, dtype=str)

    db.columns = db.columns.str.lower().str.strip()

    db = db.fillna("")

    required = [
        "brand_name",
        "generic_name",
        "strength in pad"
    ]

    return db[required]


def find_best_match(ocr_text, db):

    clean_text = normalize(ocr_text)

    words = clean_text.split()

    strengths = extract_strength(ocr_text)

    best_score = 0
    best_row = None

    for _, row in db.iterrows():

        brand = normalize(row["brand_name"])
        generic = normalize(row["generic_name"])

        brand_score = fuzz.partial_ratio(
            clean_text,
            brand
        ) / 100

        generic_score = fuzz.partial_ratio(
            clean_text,
            generic
        ) / 100

        name_score = max(
            brand_score,
            generic_score
        )

        if name_score < 0.70:
            continue

        db_strength = str(
            row["strength in pad"]
        ).strip().upper()

        strength_score = 0.6

        if db_strength:

            db_strength = db_strength.replace(" ", "")

            if db_strength in strengths:
                strength_score = 1.0

        final_score = (
            0.85 * name_score +
            0.15 * strength_score
        )

        if final_score > best_score:
            best_score = final_score
            best_row = row

    if best_row is None:

        return {
            "brand_name": "unknown",
            "generic_name": "unknown",
            "strength": "unknown",
            "confidence": 0.0
        }

    return {
        "brand_name": best_row["brand_name"],
        "generic_name": best_row["generic_name"],
        "strength": best_row["strength in pad"],
        "confidence": round(best_score, 2)
    }


# ---------------- Example ----------------

if __name__ == "__main__":

    db = load_database("medicine_database.xlsx")

    ocr_text = """
    DOLO 650
    PARACETAMOL TABLETS IP
    MICRO LABS LTD
    """

    result = find_best_match(
        ocr_text,
        db
    )

    print(result)
