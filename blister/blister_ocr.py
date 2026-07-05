from ocr.inference import cnn_bilstm_ctc_ocr


def extract_blister_text(image_path):
    """
    Extract OCR text from a blister image using the
    CNN + BiLSTM + CTC OCR pipeline.
    """

    text, confidence = cnn_bilstm_ctc_ocr(image_path)

    return {
        "ocr_text": text,
        "ocr_confidence": confidence
    }
