from src.models.predict import (
    predict_disease
)


def test_prediction():

    symptoms = (
        "fever and headache"
    )

    prediction = predict_disease(
        symptoms
    )

    assert isinstance(
        prediction,
        str
    )

    assert len(prediction) > 0