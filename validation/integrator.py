from utils.cleaners import normalize

import re

DOSAGE_FORMS = [
    "tab", "tablet",
    "cap", "capsule",
    "syr", "syrup",
    "inj", "injection"
]

def normalize(text):
    text = text.lower()
    for form in DOSAGE_FORMS:
        text = text.replace(form, "")
    return re.sub(r'[^a-z0-9]', '', text)


def generic_match(presc_name, blister_generic):
    p = normalize(presc_name)
    g = normalize(blister_generic)

    print("Normalized after cleaning:", p, g)

    # direct containment
    if p in g or g in p:
        print("MATCH: containment")
        return True

    # stem matching (DOXY → DOXYCYCLINE)
    if len(p) >= 4 and g.startswith(p):
        print("MATCH: stem")
        return True
    return False


    if len(p) >= 4 and g.startswith(p):
        print("MATCH: prefix")
        return True

    if len(g) >= 4 and p.startswith(g):
        print("MATCH: reverse prefix")
        return True

    print("NO MATCH")
    return False




def integrate_prescription_and_blisters(prescription, blisters):
    validated = []
    used_blisters = set()
    unmatched_prescription = []

    for sched in prescription["schedules"]:
        matched = False

        for i, blister in enumerate(blisters):
            if i in used_blisters:
                continue

            if blister["confidence"] < 0.6:
                continue

            if generic_match(sched["medicine_name"], blister["generic_name"]):
                validated.append({
                    "brand_name": blister["brand_name"],
                    "generic_name": blister["generic_name"],
                    "strength": blister["strength"],
                    "frequency": sched["frequency"],
                    "duration": sched["duration"],
                    "food_relation": sched["food_relation"],
                    "final_confidence": min(
                        sched["confidence"], blister["confidence"]
                    )
                })
                used_blisters.add(i)
                matched = True
                break

        if not matched:
            unmatched_prescription.append(sched)

    unmatched_blisters = [
        b for i, b in enumerate(blisters) if i not in used_blisters
    ]

    return {
        "validated_medicines": validated,
        "unmatched_prescription_entries": unmatched_prescription,
        "unmatched_blisters": unmatched_blisters,
        "needs_manual_review": (
            len(unmatched_prescription) > 0 or len(unmatched_blisters) > 0
        )
    }
