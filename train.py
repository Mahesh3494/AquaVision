import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ================================
# STEP 1 — CONFIGURATION
# ================================
dataset_path = r"C:\aquavision\data\Augmented_shrimp"
IMG_SIZE = 224  # EfficientNet expects 224x224
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 4

# ================================
# STEP 2 — LOAD AND PREPARE DATA
# ================================
# No heavy augmentation this time — just mild adjustments
# EfficientNet is already powerful enough, doesn't need aggressive augmentation

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print("\nClass mapping:", train_data.class_indices)

# ================================
# STEP 3 — BUILD THE MODEL
# ================================
# WHY TRANSFER LEARNING WORKS:
# EfficientNetB0 was trained on ImageNet — 1.2 million images, 1000 categories
# It already learned to detect edges, textures, shapes, color patterns
# We FREEZE those layers (don't change them) and only train our new top layers
# Our top layers learn: "these specific textures/colors = shrimp disease X"

# Load EfficientNetB0 without the top classification layer
base_model = EfficientNetB0(
    weights='imagenet',        # use pretrained weights
    include_top=False,         # remove the last layer (was for 1000 ImageNet classes)
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze the base model — don't touch the pretrained weights yet
base_model.trainable = False

# Add our own classification layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)    # converts feature maps to single vector
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)  # 4 classes

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"\nTotal layers: {len(model.layers)}")
print(f"Trainable layers: {len([l for l in model.layers if l.trainable])}")

# ================================
# STEP 4 — CALLBACKS
# ================================
# EarlyStopping = stops training automatically if val accuracy stops improving
# Saves you from waiting 20 epochs when model already peaked at epoch 12

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=4,            # stop if no improvement for 4 consecutive epochs
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    r"C:\aquavision\best_model.h5",
    monitor='val_accuracy',
    save_best_only=True,   # only saves when val accuracy improves
    verbose=1
)

# ================================
# STEP 5 — PHASE 1 TRAINING
# ================================
# Train only our new top layers first (base model frozen)
print("\n========== PHASE 1: Training top layers ==========")

history1 = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data,
    callbacks=[early_stop, checkpoint]
)

# ================================
# STEP 6 — PHASE 2 FINE TUNING
# ================================
# Now unfreeze the last 20 layers of EfficientNet
# Let them adjust slightly to shrimp disease features
# Use very low learning rate so we don't destroy pretrained weights

print("\n========== PHASE 2: Fine tuning ==========")

base_model.trainable = True

# Freeze all layers except last 20
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Recompile with much lower learning rate
model.compile(
    optimizer=Adam(learning_rate=0.0001),  # 10x lower than before
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_data,
    epochs=10,
    validation_data=val_data,
    callbacks=[early_stop, checkpoint]
)

# ================================
# STEP 7 — PLOT RESULTS
# ================================
# Combine both phases for plotting
acc = history1.history['accuracy'] + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(acc, label='Train Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.axvline(x=len(history1.history['accuracy']), color='gray',
            linestyle='--', label='Fine tuning start')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.axvline(x=len(history1.history['loss']), color='gray',
            linestyle='--', label='Fine tuning start')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig(r"C:\aquavision\efficientnet_results.png")
plt.show()

# ================================
# STEP 8 — SAVE FINAL MODEL
# ================================
model.save(r"C:\aquavision\shrimp_efficientnet_final.h5")
print("\nFinal model saved!")
print(f"\nBest validation accuracy: {max(val_acc):.4f}")