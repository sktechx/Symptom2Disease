from src.features.entities import (
    create_entity_recognizer,
    extract_entities,
    extract_symptoms
)


def main():

    nlp = create_entity_recognizer()

    sentences = [

        "I have fever and headache.",

        "I am suffering from cough and sore throat.",

        "I have chest pain and difficulty breathing.",

        "I have nausea, vomiting and stomach pain.",

        "I am feeling fatigue, weakness and dizziness.",

        "I have skin rash and itching."
    ]

    for text in sentences:

        print(
            "\n" + "=" * 60
        )

        print(
            "Text:",
            text
        )

        entities = extract_entities(
            text,
            nlp
        )

        print(
            "\nEntities:"
        )

        for entity in entities:

            print(
                f"{entity['text']} "
                f"→ "
                f"{entity['label']}"
            )

        symptoms = extract_symptoms(
            text,
            nlp
        )

        print(
            "\nSymptoms:",
            symptoms
        )


if __name__ == "__main__":

    main()