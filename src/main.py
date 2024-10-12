from dataset_loader import preprocess_data, augment_data

# Step 2: Dataset Collection and Preprocessing
X_train, X_test, y_train, y_test = preprocess_data()

# Perform Data Augmentation on the training set
datagen = augment_data(X_train, y_train)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
