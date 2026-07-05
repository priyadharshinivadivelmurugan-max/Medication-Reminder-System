from ocr.inference import cnn_bilstm_ctc_ocr

from llm.interpreter import interpret_prescription
from llm.blister_interpreter import interpret_blister

from validation.integrator import integrate_prescription_and_blisters

from scheduler.scheduler import generate_schedule

from utils.dosage_form import infer_dosage_form


def clean_blister_output(blister):

    return {

        "brand_name": blister.get("brand_name", "unknown"),

        "generic_name": blister.get("generic_name", "unknown"),

        "strength": blister.get("strength", "unknown"),

        "confidence": blister.get("confidence", 0.0)

    }


def run_pipeline(
        prescription_image_path,
        blister_image_paths
):

    #######################################################
    # STEP 1
    # Prescription OCR
    #######################################################

    pres_text, _ = cnn_bilstm_ctc_ocr(
        prescription_image_path
    )

    #######################################################
    # STEP 2
    # Prescription Interpretation
    #######################################################

    prescription = interpret_prescription(
        pres_text
    )

    for medicine in prescription["schedules"]:

        if (

            not medicine.get("dosage_form")

            or

            medicine["dosage_form"] == "unknown"

        ):

            medicine["dosage_form"] = infer_dosage_form(

                medicine["medicine_name"]

            )

    original_schedules = list(

        prescription["schedules"]

    )

    #######################################################
    # STEP 3
    # Blister OCR
    #######################################################

    blister_results = []

    for image in blister_image_paths:

        text, _ = cnn_bilstm_ctc_ocr(image)

        blister = interpret_blister(text)

        blister_results.append(

            clean_blister_output(blister)

        )

    #######################################################
    # STEP 4
    # Tablet Validation
    #######################################################

    tablet_prescription = [

        medicine

        for medicine in original_schedules

        if medicine["dosage_form"]

        in [

            "tablet",

            "capsule"

        ]

    ]

    validation_result = {

        "tablet_validation_done": True,

        "matched": [],

        "unmatched_prescription": [],

        "unmatched_blisters": [],

        "needs_manual_review": False

    }

    validated_schedule = original_schedules

    if tablet_prescription:

        validation = integrate_prescription_and_blisters(

            {

                "schedules": tablet_prescription

            },

            blister_results

        )

        validation_result = {

            "tablet_validation_done": True,

            "matched": validation["validated_medicines"],

            "unmatched_prescription": validation["unmatched_prescription_entries"],

            "unmatched_blisters": validation["unmatched_blisters"],

            "needs_manual_review": validation["needs_manual_review"]

        }

        if validation["validated_medicines"]:

            validated_schedule = validation["validated_medicines"]

    #######################################################
    # STEP 5
    # Schedule Generation
    #######################################################

    schedule = generate_schedule(

        validated_schedule

    )

    #######################################################
    # FINAL OUTPUT
    #######################################################

    return {

        "schedule": schedule,

        "validation": validation_result

    }
