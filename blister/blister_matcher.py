import pandas as pd
import re
from difflib import SequenceMatcher


def normalize(text):
    """
    Normalize OCR text by removing punctuation
    and converting to lowercase.
    """
    return re.sub(r'[^a-z0-9]', '', str(text).lower())


def extract_strength(text):
    """
    Extract medicine strength from OCR text.

    Supports:
        500mg
        500 MG
        500 Mg
        0.5 g
        5 ml
        250 mcg
    """

    pattern = r'(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml)'

    matches = re.findall(
        pattern,
        text.lower()
    )

    return {
        f"{value}{unit}"
        for value, unit in matches
    }


def load_database(db_path):

    db = pd.read_excel(
        db_path,
        dtype=str
    )

    db.columns = db.columns.str.lower().str.strip()

    db = db.fillna("")

    return db[
        [
            "brand_name",
            "generic_name",
            "strength in pad"
        ]
    ]


def find_best_match(ocr_text, db):

    words = [

        normalize(word)

        for word in ocr_text.split()

        if len(word) > 2

    ]

    strengths_in_ocr = extract_strength(
        ocr_text
    )

    best_score = 0

    best_row = None

    for _, row in db.iterrows():

        brand = normalize(
            row["brand_name"]
        )

        db_strength = normalize(
            row["strength in pad"]
        )

        ###############################################
        # Brand Similarity
        ###############################################

        scores = [

            SequenceMatcher(

                None,

                brand,

                word

            ).ratio()

            for word in words

        ]

        brand_score = max(scores) if scores else 0

        if brand_score < 0.70:
            continue

        ###############################################
        # Strength Similarity
        ###############################################

        strength_score = 0.6

        if db_strength:

            if db_strength in strengths_in_ocr:

                strength_score = 1.0

        ###############################################
        # Final Score
        ###############################################

        final_score = (

            0.85 * brand_score +

            0.15 * strength_score

        )

        if final_score > best_score:

            best_score = final_score

            best_row = row

    if best_row is None:

        return None

    return {

        "brand_name": best_row["brand_name"],

        "generic_name": best_row["generic_name"],

        "strength": best_row["strength in pad"],

        "confidence": round(best_score, 2)

    }
