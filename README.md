# AutoJudge: Predicting Programming Problem Difficulty

## Project Overview
Online coding platforms classify programming problems into difficulty levels like Easy, Medium, and Hard, and also assign a numerical difficulty score.  
This categorization is generally based on human judgment, user feedback, and historical solve data.

In this project, we build an automated system called **AutoJudge** that predicts:
- Problem difficulty class (Easy / Medium / Hard) – Classification
- Problem difficulty score – Regression

The prediction is done using only the textual content of a programming problem.

---

## Dataset Used
The dataset contains programming problems with the following fields:
- `title`
- `description`
- `input_description`
- `output_description`
- `problem_class` (Easy / Medium / Hard)
- `problem_score` (numerical value)

---

## Approach

### 1. Data Preprocessing
- Combined all text fields (title, description, input, output) into a single text column
- Removed duplicates
- Converted text to lowercase and cleaned unnecessary characters

---

### 2. Feature Extraction
The following features were used:
- TF-IDF vectors of combined text
- Text length
- Number of mathematical symbols
- Keyword presence (e.g., dp, graph, recursion)
- number of unique words and word count 

---

## Models Used

### Classification Models
Used to predict difficulty class:
- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)

### Regression Models
Used to predict numerical difficulty score:
- Linear Regression
- Ridge Regression
- Gradient Boosting Regressor

---

## Evaluation Metrics

### Classification
- Accuracy
- Confusion Matrix
- Precision, Recall, F1-score

### Regression
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## Web Interface
A simple web interface is built using Streamlit/Flask that allows users to:
- Paste problem description
- Paste input description
- Paste output description
- Click a Predict button

The interface displays:
- Predicted difficulty class
- Predicted difficulty score

---

## Saved Models
The trained models are saved locally using pickle:
- Classification model
- Regression model

These models are loaded directly in the web interface for prediction.

---

## How to Run the Project Locally

1. Clone the repository:
```bash
   git clone https://github.com/aayushi0911/AutoJudge
```
Install required libraries:
```bash
pip install -r requirements.txt
```
Run the web application:
```bash
streamlit run app.py
```

---
Author

Name: Aayushi Bhardwaj
Enrollment No.: 23112001
