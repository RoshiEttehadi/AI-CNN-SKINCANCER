import tensorflow as tf
from dataset_loader import load_data
from melanoma_model import melanomaCNN
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from src.evaluate import evaluate_model


def train_model():
    # Load dataset
    X_train, X_val, y_train, y_val = load_data()

    # Create the CNN model
    model = melanomaCNN(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))

    # Compile the model (this was missing)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Set up callbacks
    checkpoint = tf.keras.callbacks.ModelCheckpoint('melanoma_model_best.keras', save_best_only=True, monitor='val_loss', mode='min')
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Define the number of epochs
    epochs = 50

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32,
                        validation_data=(X_val, y_val),
                        callbacks=[checkpoint, early_stop],
                        verbose=2)

    # Save the trained model
    model.save('melanoma_model_final.keras')
    print("Model training completed and saved!")

    # Evaluate the model
    evaluate_model(model, X_val, y_val)

    # Visualize the training result
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(5, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

if __name__ == "__main__":
    train_model()
