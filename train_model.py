import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define paths (update with your actual paths)
BASE_DIR = 'dataset/chest_xrays/'
MODEL_PATH = 'models/pulmonary_edema_model.h5'

# Create directories if they don't exist
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Image data generators with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Use 20% of the data for validation
)

# Set up image generators from directory structure
train_generator = train_datagen.flow_from_directory(
    BASE_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training',  # Use training subset
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    BASE_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation',  # Use validation subset
    shuffle=False
)

# Print the class indices to verify
print("Class indices:", train_generator.class_indices)

# Create model using transfer learning with DenseNet121
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model - FIX: Use explicitly named metrics
metrics = [
    'accuracy', 
    tf.keras.metrics.AUC(name='auc_1'),           # Fixed metric name
    tf.keras.metrics.Recall(name='recall_1'),     # Fixed metric name
    tf.keras.metrics.Precision(name='precision_1') # Fixed metric name
]

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=metrics
)

# Model callbacks - FIX: Change monitor from 'val_auc' to 'val_auc_1'
callbacks = [
    ModelCheckpoint(MODEL_PATH, monitor='val_auc_1', mode='max', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_auc_1', patience=10, mode='max', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
]

# Train the model
history = model.fit(
    train_generator,
    epochs=15,
    validation_data=validation_generator,
    callbacks=callbacks
)

# Load the best model
model = tf.keras.models.load_model(MODEL_PATH)

# Fine-tune the model by unfreezing some layers
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Recompile with a lower learning rate - use same named metrics
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=metrics
)

# Continue training
history_fine = model.fit(
    train_generator,
    epochs=15,
    validation_data=validation_generator,
    callbacks=callbacks
)

# Save the final model
model.save('models/pulmonary_edema_model_final.h5')

# Plot training history
plt.figure(figsize=(12, 8))

# Plot accuracy
plt.subplot(2, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

# Plot loss
plt.subplot(2, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

# FIX: Update the history plot keys to match the new metric names
# Plot AUC
plt.subplot(2, 2, 3)
plt.plot(history.history['auc_1'])
plt.plot(history.history['val_auc_1'])
plt.title('Model AUC')
plt.ylabel('AUC')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

# Plot Recall
plt.subplot(2, 2, 4)
plt.plot(history.history['recall_1'])
plt.plot(history.history['val_recall_1'])
plt.title('Model Recall')
plt.ylabel('Recall')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# Evaluate the model on the validation data
print("\nEvaluating model on validation data:")
validation_results = model.evaluate(validation_generator)
print("Validation Loss: {:.4f}".format(validation_results[0]))
print("Validation Accuracy: {:.4f}".format(validation_results[1]))
print("Validation AUC: {:.4f}".format(validation_results[2]))
print("Validation Recall: {:.4f}".format(validation_results[3]))
print("Validation Precision: {:.4f}".format(validation_results[4]))

print("Model training complete!")
