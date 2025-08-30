import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

class GesturePredictor:
    def __init__(self, model_path, labels_path, seq_len=32, use_z=False):
        self.model = tf.keras.models.load_model(model_path)
        self.seq_len = seq_len
        self.use_z = use_z
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        self.labels = self._load_labels(labels_path)

    def __del__(self):
        self.hands.close()

    def _load_labels(self, labels_path):
        labels = {}
        with open(labels_path, 'r', encoding='utf-8') as f:
            for line in f:
                idx, name = line.strip().split(',')
                labels[int(idx)] = name
        return labels

    def _normalize_landmarks(self, landmarks, img_w, img_h):
        pts = []
        for lm in landmarks:
            x = lm.x * img_w
            y = lm.y * img_h
            if self.use_z:
                z = lm.z * max(img_w, img_h)
                pts.append([x, y, z])
            else:
                pts.append([x, y])
        pts = np.array(pts, dtype=np.float32)

        origin = pts[0].copy()
        pts -= origin

        scale = (np.abs(pts[:, :2]).max() + 1e-6)
        pts[:, :2] /= scale
        if self.use_z:
            pts[:, 2] /= scale

        if pts[17, 0] > pts[5, 0]:
            pts[:, 0] *= -1.0

        return pts.flatten()

    def predict_on_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Không thể mở video: {video_path}")
            return None, None
        
        sequence = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.hands.process(rgb)
            
            if res.multi_hand_landmarks:
                hand_landmarks = res.multi_hand_landmarks[0]
                normalized_keypoints = self._normalize_landmarks(hand_landmarks.landmark, w, h)
                sequence.append(normalized_keypoints)
            else:
                sequence.append(np.zeros((21 * (2 + self.use_z),), dtype=np.float32))

            if len(sequence) == self.seq_len:
                break

        cap.release()

        if len(sequence) < self.seq_len:
            if len(sequence) > 0:
                pad_len = self.seq_len - len(sequence)
                last_frame = sequence[-1]
                sequence.extend([last_frame] * pad_len)
            else:
                sequence = [np.zeros((21 * (2 + self.use_z),), dtype=np.float32)] * self.seq_len

        sequence = np.array(sequence, dtype=np.float32)
        input_data = np.expand_dims(sequence, axis=0)

        y_pred_prob = self.model.predict(input_data)
        y_pred_class = np.argmax(y_pred_prob, axis=1)[0]
        
        return y_pred_class, y_pred_prob[0]

    def process_and_export_video(self, input_path, output_path):

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Không thể mở video: {input_path}")
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None
        
        sequence = []
        prediction_label = "Loading"
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            
            if out is None:
                fps = cap.get(cv2.CAP_PROP_FPS)
                out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.hands.process(rgb)
            
            if res.multi_hand_landmarks:
                hand_landmarks = res.multi_hand_landmarks[0]
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                normalized_keypoints = self._normalize_landmarks(hand_landmarks.landmark, w, h)
                sequence.append(normalized_keypoints)
            else:
                sequence.append(np.zeros((21 * 2,), dtype=np.float32))

            if len(sequence) == self.seq_len:
                input_data = np.expand_dims(np.array(sequence, dtype=np.float32), axis=0)
                y_pred_prob = self.model.predict(input_data, verbose=0)
                y_pred_class = np.argmax(y_pred_prob, axis=1)[0]
                prediction_label = self.labels.get(y_pred_class, "Không xác định")
            
                sequence = []

            cv2.putText(frame, f"Predict: {prediction_label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            out.write(frame)

        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        print(f"Đã xuất video xử lý tại: {output_path}")

if __name__ == '__main__':
    predictor = GesturePredictor(
        model_path="gesture_train_kp_dataset/seq_classifier.keras",
        labels_path="gesture_train_kp_dataset/labels.csv"
    )

    input = input()
    output = "processed_video.mp4"
    print(predictor.predict_on_video(input))
    predictor.process_and_export_video(input, output)
    