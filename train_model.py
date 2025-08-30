# train_seq_model.py
import numpy as np, tensorflow as tf, os
from sklearn.utils.class_weight import compute_class_weight

DATA_DIR = "gesture_train_kp_dataset"
num_epochs = 200
batch_size = 64
lr = 1e-3

X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))

num_classes = len(open(os.path.join(DATA_DIR,"labels.csv")).read().strip().splitlines())
SEQ_LEN = X_train.shape[1]
FEAT_DIM = X_train.shape[2]

print("Train shape:", X_train.shape, y_train.shape)
print("Val shape:", X_val.shape, y_val.shape)

inputs = tf.keras.layers.Input(shape=(SEQ_LEN, FEAT_DIM))
x = tf.keras.layers.Masking(mask_value=0.0)(inputs)
x = tf.keras.layers.LSTM(128, return_sequences=True)(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.LSTM(64)(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(lr),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()


classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
cw = {int(c): float(w) for c, w in zip(classes, class_weights)}

cw[1] = cw[1]/4
cw[5] = cw[5]/3
cw[8] = cw[8]/3
print("Class weights:", cw)
cb = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=num_epochs,
    batch_size=batch_size,
    class_weight=cw,
    callbacks=cb
)

model.save(os.path.join(DATA_DIR, "seq_classifier.keras"))

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()
open(os.path.join(DATA_DIR, "seq_classifier.tflite"), "wb").write(tflite_model)

print("Saved models to", DATA_DIR)
