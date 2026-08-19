# ============================================================
# DecodeLabs Artificial Intelligence Internship
# Project 2: Data Classification Using AI
# ============================================================

# Import required libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. Load the Iris Dataset
# ------------------------------------------------------------

iris = load_iris()

print("========================================")
print("       IRIS FLOWER CLASSIFICATION")
print("========================================")

print("\nFeature Names:", iris.feature_names)
print("Target Names:", iris.target_names)
print("Data Shape:", iris.data.shape)
print("Target Shape:", iris.target.shape)


# ------------------------------------------------------------
# 2. Separate Features and Target
# ------------------------------------------------------------

X = iris.data
y = iris.target


# ------------------------------------------------------------
# 3. Split Data into Training and Testing Sets
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# ------------------------------------------------------------
# 4. Create and Train the KNN Classification Model
# ------------------------------------------------------------

model = KNeighborsClassifier(n_neighbors=3)

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")


# ------------------------------------------------------------
# 5. Make Predictions on Test Data
# ------------------------------------------------------------

y_pred = model.predict(X_test)

print("\nPredicted Classes:")
print(y_pred)

print("\nActual Classes:")
print(y_test)


# ------------------------------------------------------------
# 6. Evaluate Model Performance
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)

# ------------------------------------------------------------
# 7. Display Confusion Matrix
# ------------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

display.plot()

plt.title("Iris Classification - Confusion Matrix")
plt.show()
# ------------------------------------------------------------
# 8. Predict the Species of a New Flower
# ------------------------------------------------------------

new_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(new_flower)

predicted_species = iris.target_names[prediction[0]]

print("\nNew Flower Measurements:", new_flower[0])
print("Predicted Species:", predicted_species)

print("\n========================================")
print("       CLASSIFICATION COMPLETED")
print("========================================")