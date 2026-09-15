from src.data.preprocess import (
    preprocess_text
)


def test_lowercase():

    result = preprocess_text(
        "Fever and Headache"
    )

    assert result == (
        "fever and headache"
    )


def test_special_characters():

    result = preprocess_text(
        "Fever!!! Headache@123"
    )

    assert "!" not in result
    assert "@" not in result


def test_extra_spaces():

    result = preprocess_text(
        "fever     headache"
    )

    assert result == (
        "fever headache"
    )