from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import tensorflow as tf

DATA_DIR = "gesture_train_kp_dataset"
model = tf.keras.models.load_model(DATA_DIR + "/seq_classifier.keras")
X_test = np.load(DATA_DIR + "/X_test.npy")
y_test = np.load(DATA_DIR + "/y_test.npy")

y_pred = np.argmax(model.predict(X_test), axis=1)

print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print(cm)
