# 💊 AI-Powered Prescription & Medication Management System

> An intelligent healthcare assistance system that automatically extracts medicine information from **tablet blister packs**, **printed prescriptions**, and **handwritten prescriptions** using a custom OCR pipeline based on **CRAFT Text Detection** and **CNN + BiLSTM + CTC Recognition**, followed by NLP-based medicine interpretation, validation, and automated medication schedule generation.

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Neural%20Networks-red?logo=keras)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20Application-ff4b4b?logo=streamlit)
![OCR](https://img.shields.io/badge/OCR-CRAFT%20%7C%20CRNN-success)

</p>

---

# 📖 Overview

Medication adherence is a significant challenge, particularly for elderly individuals, patients with chronic illnesses, and people who struggle to interpret handwritten prescriptions or medicine packaging. Traditional reminder applications depend on manual data entry and are unable to automatically understand medicine information from prescriptions or tablet blister packs.

This project presents an **AI-powered Medication Management System** capable of automatically extracting medicine names, dosage instructions, and medication schedules directly from medical images.

Unlike conventional OCR applications, the proposed system combines **Computer Vision**, **Deep Learning**, and **Natural Language Processing** into a unified pipeline capable of understanding both structured and unstructured medical documents.

The extracted information is validated against a medicine database, interpreted into meaningful dosage instructions, and transformed into an automated medication schedule with reminder support.

---

# ✨ Key Features

- 💊 Tablet blister pack text extraction
- 📄 Printed prescription recognition
- ✍️ Handwritten prescription recognition
- 🔍 Character-level text detection using **CRAFT**
- 🧠 Custom OCR using **CNN + BiLSTM + CTC**
- 📚 Medicine database validation
- 📝 Intelligent dosage interpretation
- 📅 Automatic medication schedule generation
- 🔔 Reminder scheduling
- 🖥️ Interactive Streamlit application
- ⚡ End-to-end automated medication pipeline

---

# 🏗️ System Architecture

```text
                  Input Image
                       │
                       ▼
             Image Preprocessing
                       │
                       ▼
          CRAFT Text Detection Model
                       │
                       ▼
        CNN + BiLSTM + CTC OCR Model
                       │
                       ▼
        Medicine Information Extraction
                       │
                       ▼
        NLP-based Interpretation Module
                       │
                       ▼
      Medicine Database Validation
                       │
                       ▼
     Medication Schedule Generation
                       │
                       ▼
             Reminder Scheduler
```

---

# 🔬 Methodology

The proposed methodology follows a multi-stage intelligent document understanding pipeline specifically designed for healthcare applications.

## 1. Data Collection

The system was developed using multiple datasets consisting of:

- Tablet blister pack images
- Printed prescriptions
- Handwritten prescriptions
- Medicine metadata database

The collected datasets contain significant variations in:

- Font styles
- Handwriting patterns
- Lighting conditions
- Camera angles
- Foil reflections
- Text orientation

making the system robust for real-world medical documents.

---

## 2. Image Preprocessing

Before OCR, every image undergoes document-specific preprocessing.

### Tablet Blister Packs

- Letterbox resizing
- Normalization
- CLAHE enhancement
- Grayscale conversion
- ROI extraction
- Local deskewing
- ROI normalization

### Printed Prescriptions

- Letterbox resizing
- Grayscale conversion
- Bilateral filtering
- CLAHE
- Adaptive Gaussian Thresholding

### Handwritten Prescriptions

- All printed prescription preprocessing
- Morphological stroke repair
- Handwritten ROI extraction

These preprocessing stages significantly improve OCR robustness under varying environmental conditions.

---

## 3. OCR Pipeline

Unlike traditional OCR systems, the proposed approach separates text detection and text recognition into two dedicated deep learning stages.

### Character Detection

**CRAFT (Character Region Awareness for Text Detection)**

CRAFT identifies individual character regions and their spatial relationships, enabling accurate detection of curved, rotated, and irregular medical text.

### Text Recognition

A custom **CNN + BiLSTM + CTC** architecture performs sequence-based text recognition.

The recognition network consists of:

- Deep CNN feature extractor
- Bidirectional LSTM sequence modeling
- Connectionist Temporal Classification (CTC) decoding

This architecture effectively recognizes:

- Printed prescriptions
- Handwritten prescriptions
- Tablet blister pack text

without requiring explicit character segmentation.

---

## 4. Medicine Interpretation

After OCR, the extracted text undergoes domain-specific interpretation.

The system:

- Cleans OCR output
- Identifies medicine names
- Detects dosage patterns
- Extracts medicine frequency
- Interprets administration instructions

The interpreted information is then validated using a structured medicine database.

---

## 5. Medication Schedule Generation

The validated medicine information is automatically converted into a structured medication schedule.

The scheduler determines:

- Medicine name
- Dosage
- Administration frequency
- Daily schedule

The generated schedule serves as the basis for reminder notifications.

---

# 📂 Project Structure

```text
PRESCRIPTION_MEDICATION_SYSTEM/
│
├── app.py
├── requirements.txt
├── README.md
│
├── dataset/
├── blister/
├── llm/
├── models/
├── ocr/
├── output/
├── pipeline/
├── scheduler/
├── training/
├── utils/
└── validation/
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/<username>/PRESCRIPTION_MEDICATION_SYSTEM.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

The application allows users to upload:

- Tablet blister pack images
- Printed prescriptions
- Handwritten prescriptions

and automatically generates structured medication information. Followed by this run the following command

```bash
streamlit run alarm_engine.py
```

---

# 🧠 Deep Learning Models

| Model | Purpose |
|--------|----------|
| CRAFT | Character-level text detection |
| CNN | Visual feature extraction |
| BiLSTM | Sequential text modeling |
| CTC Decoder | Character sequence prediction |

---

# 📊 Supported Inputs

- Tablet blister packs
- Printed prescriptions
- Handwritten prescriptions

---

# 🛠️ Technologies Used

## Programming

- Python

## Computer Vision

- OpenCV

## Deep Learning

- TensorFlow
- Keras
- CNN
- BiLSTM
- CTC

## OCR

- CRAFT Text Detector
- Custom OCR Pipeline

## NLP

- Medicine interpretation
- Dosage extraction
- Medicine validation

## Interface

- Streamlit

---

# 🔮 Future Enhancements

- Mobile application support
- Cloud deployment
- Voice-based medication reminders
- Multilingual prescription recognition
- Electronic Health Record (EHR) integration
- Advanced transformer-based medical NLP

---

# 👩‍💻 Author

**Priyadharshini V**

Electronics and Communication Engineer

Artificial Intelligence • Computer Vision • Deep Learning • Healthcare AI

---

# 📜 License

This project was developed as part of an academic Final Year Project and is intended for educational and research purposes.
