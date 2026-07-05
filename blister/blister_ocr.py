from ocr.google_vision_ocr import google_vision_ocr_with_confidence

def extract_blister_text(image_path):
    text, conf = ocr_with_confidence(image_path)
    return {
        "ocr_text": text,
        "ocr_confidence": conf
    }

