import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator


data_dir = "/content/drive/MyDrive/CCSN_prepared/train"


filepaths = []
labels = []

for class_name in sorted(os.listdir(data_dir)):
    class_path = os.path.join(data_dir, class_name)
    if os.path.isdir(class_path):
        for fname in os.listdir(class_path):
            if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                filepaths.append(os.path.join(class_name, fname))
                labels.append(class_name)

df = pd.DataFrame({
    'filename': filepaths,
    'class': labels
})

print(f"Ukupno slika: {len(df)} u {df['class'].nunique()} klasa")


num_classes = df['class'].nunique()
input_shape = (224, 224, 3)
batch_size = 32
epochs_stage1 = 15
epochs_stage2 = 5


datagen = ImageDataGenerator(rescale=1./255)


def create_model(trainable_layers=0, lr=1e-3):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
   
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False
    for layer in base_model.layers[-trainable_layers:]:
        layer.trainable = True

    x = GlobalAveragePooling2D()(base_model.output)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1

acc_per_fold = []
loss_per_fold = []

train_acc_all, val_acc_all, train_loss_all, val_loss_all = [], [], [], []

for train_idx, val_idx in kf.split(df):
    print(f"\n===== Fold {fold} =====")

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    
    train_gen = datagen.flow_from_dataframe(
        train_df,
        directory=data_dir,
        x_col='filename',
        y_col='class',
        target_size=(224, 224),
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True
    )

    val_gen = datagen.flow_from_dataframe(
        val_df,
        directory=data_dir,
        x_col='filename',
        y_col='class',
        target_size=(224, 224),
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False
    )

    # STAGE 1
    model = create_model(trainable_layers=0, lr=1e-3)
    callbacks = [
        ModelCheckpoint(f'effnetb0_fold{fold}.h5', save_best_only=True, monitor='val_accuracy', mode='max'),
        EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)
    ]

    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs_stage1,
        callbacks=callbacks,
        verbose=1
    )

    # STAGE 2 (fine-tuning)
    model = create_model(trainable_layers=20, lr=1e-4)
    model.load_weights(f'effnetb0_fold{fold}.h5')

    callbacks_ft = [
        ModelCheckpoint(f'effnetb0_finetune{fold}.h5', save_best_only=True, monitor='val_accuracy', mode='max'),
        EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)
    ]

    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs_stage2,
        callbacks=callbacks_ft,
        verbose=1
    )

    
    train_acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    train_loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']

    train_acc_all.append(np.array(train_acc))
    val_acc_all.append(np.array(val_acc))
    train_loss_all.append(np.array(train_loss))
    val_loss_all.append(np.array(val_loss))

    scores = model.evaluate(val_gen, verbose=0)
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    fold += 1


min_len = min(len(x) for x in train_acc_all)
train_acc_all = [x[:min_len] for x in train_acc_all]
val_acc_all = [x[:min_len] for x in val_acc_all]
train_loss_all = [x[:min_len] for x in train_loss_all]
val_loss_all = [x[:min_len] for x in val_loss_all]

avg_train_acc = np.mean(train_acc_all, axis=0)
avg_val_acc = np.mean(val_acc_all, axis=0)
avg_train_loss = np.mean(train_loss_all, axis=0)
avg_val_loss = np.mean(val_loss_all, axis=0)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(avg_train_acc, label='Train Accuracy')
plt.plot(avg_val_acc, label='Validation Accuracy')
plt.title('Accuracy tokom epoha')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(avg_train_loss, label='Train Loss')
plt.plot(avg_val_loss, label='Validation Loss')
plt.title('Loss tokom epoha')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

