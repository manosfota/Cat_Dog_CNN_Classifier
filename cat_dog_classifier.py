import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ============================================
# STEP 1: LOAD THE DATA
# ============================================
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# STEP 2: FILTER ONLY CATS (3) AND DOGS (5)
train_filter = (y_train == 3) | (y_train == 5)
test_filter = (y_test == 3) | (y_test == 5)

x_train = x_train[train_filter.flatten()]
y_train = y_train[train_filter.flatten()]
x_test = x_test[test_filter.flatten()]
y_test = y_test[test_filter.flatten()]

# STEP 3: CONVERT LABELS — CAT=0, DOG=1
y_train = (y_train == 5).astype(int)
y_test = (y_test == 5).astype(int)

# STEP 4: NORMALIZE PIXEL VALUES FROM 0-255 TO 0-1
x_train = x_train / 255.0
x_test = x_test / 255.0

print(f"Training images: {x_train.shape[0]}")
print(f"Test images: {x_test.shape[0]}")
print(f"Image shape: {x_train.shape[1:]}")

# ============================================
# STEP 5: BUILD THE NEURAL NETWORK
# ============================================
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# STEP 6: COMPILE THE MODEL
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()

# ============================================
# STEP 7: TRAIN THE MODEL
# ============================================
history = model.fit(
    x_train, y_train,
    epochs=15,
    batch_size=32,
    validation_split=0.2
)

# ============================================
# STEP 8: EVALUATE THE MODEL
# ============================================
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")

# ============================================
# STEP 9: PLOT TRAINING HISTORY
# ============================================

# Accuracy plot
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
#plt.show()

# ============================================
# STEP 10: SHOW PREDICTIONS ON TEST IMAGES
# ============================================
predictions = model.predict(x_test[:16])
class_names = ['Cat', 'Dog']

plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(x_test[i])
    predicted = class_names[int(predictions[i][0] > 0.5)]
    actual = class_names[y_test[i][0]]
    color = 'green' if predicted == actual else 'red'
    plt.title(f"P:{predicted} | A:{actual}", color=color, fontsize=9)
    plt.axis('off')

plt.tight_layout()
plt.savefig('predictions.png', dpi=150)
#plt.show()

# ============================================
# STEP 11: CONFUSION MATRIX
# ============================================
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

all_predictions = (model.predict(x_test) > 0.5).astype(int)
cm = confusion_matrix(y_test, all_predictions)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Cat', 'Dog'],
            yticklabels=['Cat', 'Dog'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
#plt.show()

print("\nClassification Report:")
print(classification_report(y_test, all_predictions, target_names=['Cat', 'Dog']))
