from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np


def evaluate_model(model, X_val, y_val):
    # Evaluate the model on the validation set
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")

    # Make predictions on the validation set
    y_pred = model.predict(X_val)
    y_pred_class = np.argmax(y_pred, axis=1)

    # Calculate performance metrics
    accuracy = accuracy_score(y_val, y_pred_class)
    precision = precision_score(y_val, y_pred_class)
    recall = recall_score(y_val, y_pred_class)
    f1 = f1_score(y_val, y_pred_class)

    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall: {recall:.4f}")
    print(f"Validation F1-score: {f1:.4f}")

    # Calculate confusion matrix
    conf_mat = confusion_matrix(y_val, y_pred_class)
    print("Confusion Matrix:")
    print(conf_mat)

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_val, y_pred_class))