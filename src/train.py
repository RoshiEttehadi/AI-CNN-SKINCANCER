import tensorflow as tf
from dataset_loader import load_data
from cnn_model import create_cnn_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from src.evaluate import evaluate_model


def train_model():
    # Load dataset
    X_train, X_val, y_train, y_val = load_data()

    # Create the CNN model
    model = create_cnn_model()

    # Set up callbacks
    checkpoint = ModelCheckpoint('cnn_model_best.keras', save_best_only=True, monitor='val_loss', mode='min')
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32,
              callbacks=[checkpoint, early_stop])

    # Save the trained model
    model.save('cnn_model_final.keras')
    print("Model training completed and saved!")

    # Evaluate the model
    evaluate_model(model, X_val, y_val)

if __name__ == "__main__":
    train_model()
