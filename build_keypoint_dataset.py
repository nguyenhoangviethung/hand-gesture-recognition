import os, glob, cv2, math, numpy as np
import mediapipe as mp
from tqdm import tqdm
from sklearn.model_selection import train_test_split

DATA_ROOT='gesture_train_data'
OUT_DIR='gesture_train_kp_dataset_z'
SEQ_LEN=32
FRAME_STRIDE=2
MIN_DETECTION_CONF=0.5
MAX_NUM_HANDS=2
USE_Z=False
RANDOM_STATE=42

os.makedirs(OUT_DIR, exist_ok=True)
mp_hands = mp.solutions.hands

VID_EXTS=set(['.mp4', '.avi', '.mov', '.mkv'])

def list_video(folder):
    res = []
    for ext in VID_EXTS:
        res += glob.glob(os.path.join(folder,f'*{ext}'))
    return res

def normalize_landmarks(landmarks, img_w, img_h, use_z=USE_Z, mirror_left=True, align=False):
    pts = []
    for lm in landmarks:
        x = lm.x*img_w
        y = lm.y*img_h
        if use_z:
            z = lm.z*max(img_w, img_h)
            pts.append([x,y,z])
        else:
            pts.append([x,y])
    
    pts = np.array(pts, dtype=np.float32)

    origin = pts[0].copy()
    pts -= origin

    scale = (np.abs(pts[:, :2]).max() + 1e-6)   
    pts[:, :2] /= scale
    if use_z:
        pts[:, 2] /= scale
    
    if mirror_left:
        if pts[17,0] > pts[5,0]:
            pts[:,0] *= -1.0
    
    return pts.flatten()

classes = sorted([d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))])
label2id = {c : i for i, c in enumerate(classes)}

print(classes)

test_size = 0.15
val_size = 0.15
train_videos = []
val_videos = []
test_videos = []

for cls in classes:
    cls_dir = os.path.join(DATA_ROOT, cls)
    vids = list_video(cls_dir)

    if len(vids) == 0:
        continue
    
    vids_sorted = sorted(vids)
    train, test = train_test_split(vids_sorted, test_size=test_size, random_state=RANDOM_STATE)
    train2, val = train_test_split(train, test_size=val_size/(1-test_size), random_state=RANDOM_STATE)

    train_videos += [(p,cls) for p in train2]
    test_videos += [(p, cls) for p in test]
    val_videos += [(p,cls) for p in val]

def extract_sequences(video_list, split_name):
    X_list = []
    y_list = []

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONF,
        min_tracking_confidence=0.5
    ) as hand:
        for video_path, cls in tqdm(video_list, desc=f'Extract{split_name}'):
            label = label2id[cls]
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print('Fail to open', video_path)
                continue
            frames_feats = []
            fidx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                fidx += 1
                if fidx % FRAME_STRIDE != 0:
                    continue
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = hand.process(rgb)

                if res.multi_hand_landmarks:

                    best_lm = None
                    if len(res.multi_hand_landmarks) == 1:
                        best_lm = res.multi_hand_landmarks[0]
                    else:

                        best_area = -1
                        for lm in res.multi_hand_landmarks:
                            xs = [p.x for p in lm.landmark]
                            ys = [p.y for p in lm.landmark]

                            bw = (max(xs)-min(xs))*w
                            bh = (max(ys)-min(ys))*h

                            area = bw*bh

                            if area > best_area:
                                best_lm = lm
                                best_area = area
                    feat = normalize_landmarks(best_lm.landmark, w, h)
                    frames_feats.append(feat)
                else:
                    frames_feats.append(np.zeros((21*2,), dtype = np.float32))
            cap.release()

            if len(frames_feats) == 0:
                continue
            arr = np.stack(frames_feats, axis=0) 
            F = arr.shape[0]
            if F < SEQ_LEN:

                pad_n = SEQ_LEN - F
                pad = np.tile(arr[-1:], (pad_n,1))
                arr = np.concatenate([arr, pad], axis=0)
                F = arr.shape[0]

            stride_win = SEQ_LEN  

            for start in range(0, F - SEQ_LEN + 1, stride_win):
                seg = arr[start:start+SEQ_LEN]
                X_list.append(seg)
                y_list.append(label)

    X_np = np.array(X_list, dtype=np.float32)
    y_np = np.array(y_list, dtype=np.int64)
    print(f"{split_name}: {X_np.shape}, {y_np.shape}")
    return X_np, y_np

X_train, y_train = extract_sequences(train_videos, "train")
X_val,   y_val   = extract_sequences(val_videos,   "val")
X_test,  y_test  = extract_sequences(test_videos,  "test")

np.save(os.path.join(OUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUT_DIR, "X_val.npy"), X_val)
np.save(os.path.join(OUT_DIR, "y_val.npy"), y_val)
np.save(os.path.join(OUT_DIR, "X_test.npy"), X_test)
np.save(os.path.join(OUT_DIR, "y_test.npy"), y_test)

with open(os.path.join(OUT_DIR, "labels.csv"), "w", encoding="utf-8") as f:
    for name, idx in label2id.items():
        f.write(f"{idx},{name}\n")

print("Saved dataset to", OUT_DIR)
