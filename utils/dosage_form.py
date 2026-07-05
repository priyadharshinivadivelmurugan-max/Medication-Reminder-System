def infer_dosage_form(medicine_name: str) -> str:
    name = medicine_name.lower().strip()

    if name.startswith(("tab", "tablet")):
        return "tablet"

    if name.startswith(("cap", "capsule")):
        return "capsule"

    if "syr" in name or "syrup" in name:
        return "syrup"

    if "gargle" in name or "soln" in name or "solution" in name:
        return "solution"

    if name.startswith(("inj", "injection")):
        return "injection"

    if "ointment" in name or "cream" in name or "gel" in name:
        return "ointment"

    return "unknown"
