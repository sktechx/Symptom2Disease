# Symptom → Disease NLP Classifier

A machine learning based text classification project that predicts
a disease class from user-provided symptom descriptions.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Logistic Regression
- Multinomial Naive Bayes
- Linear SVC
- Matplotlib
- Seaborn
- Joblib

## Project Pipeline

Symptoms
↓
Text Preprocessing
↓
TF-IDF Feature Extraction
↓
Machine Learning Models
↓
Model Comparison
↓
Best Model
↓
Disease Prediction

## Models

1. Logistic Regression
2. Multinomial Naive Bayes
3. Linear SVC

## Dataset

Symptom2Disease dataset.

## How to Run

Install dependencies:

pip install -r requirements.txt

Train models:

python src/train_model.py

Run prediction application:

python app/app.py

## Example

Input:

fever headache body pain

Output:

Predicted Disease:
<model prediction>

## Disclaimer

This project is intended for educational and machine learning
demonstration purposes. It should not be used as a substitute
for professional medical diagnosis.