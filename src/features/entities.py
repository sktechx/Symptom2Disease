import spacy


# ============================================================
# CREATE ENTITY RECOGNIZER
# ============================================================

def create_entity_recognizer():

    # Blank English pipeline
    nlp = spacy.blank("en")

    # Add EntityRuler
    ruler = nlp.add_pipe("entity_ruler")

    # --------------------------------------------------------
    # SYMPTOM PATTERNS
    # --------------------------------------------------------

    symptom_patterns = [

        # Fever
        {"label": "SYMPTOM", "pattern": "fever"},
        {"label": "SYMPTOM", "pattern": "high fever"},
        {"label": "SYMPTOM", "pattern": "mild fever"},

        # Pain
        {"label": "SYMPTOM", "pattern": "headache"},
        {"label": "SYMPTOM", "pattern": "severe headache"},
        {"label": "SYMPTOM", "pattern": "head pain"},
        {"label": "SYMPTOM", "pattern": "chest pain"},
        {"label": "SYMPTOM", "pattern": "stomach pain"},
        {"label": "SYMPTOM", "pattern": "abdominal pain"},
        {"label": "SYMPTOM", "pattern": "back pain"},
        {"label": "SYMPTOM", "pattern": "joint pain"},
        {"label": "SYMPTOM", "pattern": "muscle pain"},

        # Respiratory
        {"label": "SYMPTOM", "pattern": "cough"},
        {"label": "SYMPTOM", "pattern": "dry cough"},
        {"label": "SYMPTOM", "pattern": "sore throat"},
        {"label": "SYMPTOM", "pattern": "shortness of breath"},
        {"label": "SYMPTOM", "pattern": "difficulty breathing"},
        {"label": "SYMPTOM", "pattern": "breathing difficulty"},
        {"label": "SYMPTOM", "pattern": "runny nose"},
        {"label": "SYMPTOM", "pattern": "blocked nose"},
        {"label": "SYMPTOM", "pattern": "stuffy nose"},
        {"label": "SYMPTOM", "pattern": "nasal congestion"},

        # Digestive
        {"label": "SYMPTOM", "pattern": "nausea"},
        {"label": "SYMPTOM", "pattern": "vomiting"},
        {"label": "SYMPTOM", "pattern": "diarrhea"},
        {"label": "SYMPTOM", "pattern": "constipation"},
        {"label": "SYMPTOM", "pattern": "loss of appetite"},
        {"label": "SYMPTOM", "pattern": "stomach ache"},

        # General
        {"label": "SYMPTOM", "pattern": "fatigue"},
        {"label": "SYMPTOM", "pattern": "weakness"},
        {"label": "SYMPTOM", "pattern": "dizziness"},
        {"label": "SYMPTOM", "pattern": "tiredness"},
        {"label": "SYMPTOM", "pattern": "chills"},
        {"label": "SYMPTOM", "pattern": "sweating"},

        # Skin
        {"label": "SYMPTOM", "pattern": "skin rash"},
        {"label": "SYMPTOM", "pattern": "rash"},
        {"label": "SYMPTOM", "pattern": "itching"},
        {"label": "SYMPTOM", "pattern": "redness"},
        {"label": "SYMPTOM", "pattern": "swelling"},

        # Neurological
        {"label": "SYMPTOM", "pattern": "confusion"},
        {"label": "SYMPTOM", "pattern": "memory loss"},
        {"label": "SYMPTOM", "pattern": "blurred vision"},
        {"label": "SYMPTOM", "pattern": "loss of consciousness"},

        # Other
        {"label": "SYMPTOM", "pattern": "weight loss"},
        {"label": "SYMPTOM", "pattern": "weight gain"},
        {"label": "SYMPTOM", "pattern": "insomnia"},
        {"label": "SYMPTOM", "pattern": "anxiety"},
        {"label": "SYMPTOM", "pattern": "depression"},
    ]

    ruler.add_patterns(symptom_patterns)

    return nlp


# ============================================================
# EXTRACT ENTITIES FROM TEXT
# ============================================================

def extract_entities(text, nlp):

    doc = nlp(text)

    entities = []

    for ent in doc.ents:

        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char
        })

    return entities


# ============================================================
# EXTRACT ONLY SYMPTOMS
# ============================================================

def extract_symptoms(text, nlp):

    doc = nlp(text)

    symptoms = []

    for ent in doc.ents:

        if ent.label_ == "SYMPTOM":

            symptoms.append(ent.text)

    return symptoms


# ============================================================
# CONVERT ENTITIES TO TEXT
# ============================================================

def entities_to_text(text, nlp):

    symptoms = extract_symptoms(text, nlp)

    return " ".join(symptoms)