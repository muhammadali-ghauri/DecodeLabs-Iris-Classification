# Data Classification Using AI

## DecodeLabs Artificial Intelligence Internship — Project 2

### Project Overview

This project demonstrates a basic supervised machine learning classification model using the Iris dataset.

The objective is to train a machine learning model to classify Iris flowers into different species based on their physical measurements.

The project demonstrates the fundamental machine learning workflow of loading data, preparing features and labels, splitting the dataset, training a classification model, making predictions, and evaluating model performance.

---

## Dataset

The project uses the Iris dataset available through Scikit-learn.

The dataset contains 150 flower samples and four input features:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The flowers belong to three classes:

- Setosa
- Versicolor
- Virginica

---

## Machine Learning Algorithm

The project uses the **K-Nearest Neighbors (KNN)** classification algorithm.

The model was configured with:

`n_neighbors = 3`

The dataset was divided into:

- 80% Training Data
- 20% Testing Data

A fixed `random_state = 42` was used to make the train-test split reproducible.

---

## Model Evaluation

The trained model was evaluated using:

- Accuracy Score
- Precision
- Recall
- F1-Score
- Classification Report
- Confusion Matrix

For the selected train-test split, the model achieved:

**Accuracy: 100.00%**

The confusion matrix showed:

- Setosa: 10/10 correctly classified
- Versicolor: 9/9 correctly classified
- Virginica: 11/11 correctly classified

This means all 30 samples in the test set were correctly classified for this particular split.

---

## New Flower Prediction

The trained model was also tested with a new flower containing the following measurements:

- Sepal Length: 5.1 cm
- Sepal Width: 3.5 cm
- Petal Length: 1.4 cm
- Petal Width: 0.2 cm

The model predicts the corresponding Iris species based on patterns learned from the training data.

---

## Technologies Used

- Python
- Scikit-learn
- NumPy
- Matplotlib
- VS Code

---

## How to Run

1. Clone or download the project.
2. Install the required Python libraries:

`pip install numpy scikit-learn matplotlib`

3. Run the program:

`python classification.py`

---

## Learning Outcomes

Through this project, I practiced:

- Loading and understanding a dataset
- Working with features and target labels
- Supervised machine learning
- Splitting data into training and testing sets
- Training a classification model
- Making predictions on unseen data
- Evaluating model accuracy
- Understanding precision, recall, and F1-score
- Visualizing results using a confusion matrix

---

## Internship

This project was completed as **Project 2 — Data Classification Using AI** during the **DecodeLabs Artificial Intelligence Internship (Batch 2026)**.