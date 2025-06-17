# import cv2
# import mediapipe as mp
# import numpy as np
# import threading
# import pygame

# # Initialize pygame mixer for audio
# pygame.mixer.init()
# alert_sound = pygame.mixer.Sound("alert.wav")

# # Initialize mediapipe face mesh
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# # Facial landmark indices
# LEFT_EYE = [362, 385, 387, 263, 373, 380]
# RIGHT_EYE = [33, 160, 158, 133, 153, 144]
# UPPER_LIP = 13
# LOWER_LIP = 14

# # Thresholds
# EAR_THRESHOLD = 0.25
# CONSEC_FRAMES = 20
# MOUTH_OPEN_THRESHOLD = 0.05

# # Alert tracking
# counter = 0
# alert_playing = False

# def reset_alert_flag():
#     global alert_playing
#     alert_playing = False

# def play_alert():
#     global alert_playing
#     if not alert_playing:
#         alert_playing = True
#         try:
#             alert_sound.play()
#         except Exception as e:
#             print(f"[Sound Error] {e}")
#         # Reset alert flag after 2 seconds
#         threading.Timer(2.0, reset_alert_flag).start()

# def compute_EAR(landmarks, eye_indices, w, h):
#     points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
#     A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
#     B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
#     C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
#     ear = (A + B) / (2.0 * C)
#     return ear

# # Start webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     h, w, _ = frame.shape
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     result = face_mesh.process(rgb)

#     if result.multi_face_landmarks:
#         for face_landmarks in result.multi_face_landmarks:
#             # EAR calculation
#             left_ear = compute_EAR(face_landmarks.landmark, LEFT_EYE, w, h)
#             right_ear = compute_EAR(face_landmarks.landmark, RIGHT_EYE, w, h)
#             avg_ear = (left_ear + right_ear) / 2.0

#             # Mouth distance for yawn detection
#             upper_lip_y = face_landmarks.landmark[UPPER_LIP].y
#             lower_lip_y = face_landmarks.landmark[LOWER_LIP].y
#             mouth_open = abs(upper_lip_y - lower_lip_y)

#             # Drowsiness detection
#             if avg_ear < EAR_THRESHOLD:
#                 counter += 1
#                 if counter >= CONSEC_FRAMES:
#                     cv2.putText(frame, "DROWSY!", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
#                     play_alert()
#             else:
#                 counter = 0
#                 cv2.putText(frame, "Awake", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

#             # Yawning detection
#             if mouth_open > MOUTH_OPEN_THRESHOLD:
#                 cv2.putText(frame, "Yawning!", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
#                 play_alert()

#     cv2.imshow("Drowsiness + Yawning Detection", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()





# with face detection and working properly

import cv2
import mediapipe as mp
import numpy as np
import threading
import pygame
import time

# Initialize pygame mixer for audio
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("alert.wav")

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Facial landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
UPPER_LIP = 13
LOWER_LIP = 14

# Thresholds
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20
MOUTH_OPEN_THRESHOLD = 0.05

# Alert tracking
counter = 0
alert_playing = False

# Face absence tracking
no_face_start_time = None
FACE_ABSENCE_THRESHOLD = 10  # seconds

def reset_alert_flag():
    global alert_playing
    alert_playing = False

def play_alert():
    global alert_playing
    if not alert_playing:
        alert_playing = True
        try:
            alert_sound.play()
        except Exception as e:
            print(f"[Sound Error] {e}")
        # Reset alert flag after 2 seconds
        threading.Timer(2.0, reset_alert_flag).start()

def compute_EAR(landmarks, eye_indices, w, h):
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
    ear = (A + B) / (2.0 * C)
    return ear

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        no_face_start_time = None  # Reset absence timer

        for face_landmarks in result.multi_face_landmarks:
            # EAR calculation
            left_ear = compute_EAR(face_landmarks.landmark, LEFT_EYE, w, h)
            right_ear = compute_EAR(face_landmarks.landmark, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0

            # Mouth distance for yawn detection
            upper_lip_y = face_landmarks.landmark[UPPER_LIP].y
            lower_lip_y = face_landmarks.landmark[LOWER_LIP].y
            mouth_open = abs(upper_lip_y - lower_lip_y)

            # Drowsiness detection
            if avg_ear < EAR_THRESHOLD:
                counter += 1
                if counter >= CONSEC_FRAMES:
                    cv2.putText(frame, "DROWSY!", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    play_alert()
            else:
                counter = 0
                cv2.putText(frame, "Awake", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            # Yawning detection
            if mouth_open > MOUTH_OPEN_THRESHOLD:
                cv2.putText(frame, "Yawning!", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                play_alert()
    
    else:
        # Face not detected
        if no_face_start_time is None:
            no_face_start_time = time.time()
        else:
            elapsed = time.time() - no_face_start_time
            if elapsed >= FACE_ABSENCE_THRESHOLD:
                cv2.putText(frame, "FACE NOT DETECTED!", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                play_alert()

    cv2.imshow("Drowsiness + Yawning Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()









#  FPS Counter

# import cv2
# import mediapipe as mp
# import numpy as np
# import threading
# import pygame
# import time

# # Initialize pygame mixer for audio
# pygame.mixer.init()
# alert_sound = pygame.mixer.Sound("alert.wav")

# # Initialize mediapipe face mesh
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# # Facial landmark indices
# LEFT_EYE = [362, 385, 387, 263, 373, 380]
# RIGHT_EYE = [33, 160, 158, 133, 153, 144]
# UPPER_LIP = 13
# LOWER_LIP = 14

# # Thresholds
# EAR_THRESHOLD = 0.25
# CONSEC_FRAMES = 20
# MOUTH_OPEN_THRESHOLD = 0.05

# # Alert tracking
# counter = 0
# alert_playing = False

# # Face absence tracking
# no_face_start_time = None
# FACE_ABSENCE_THRESHOLD = 10  # seconds

# # FPS tracking
# prev_time = time.time()
# font = cv2.FONT_HERSHEY_SIMPLEX

# def reset_alert_flag():
#     global alert_playing
#     alert_playing = False

# def play_alert():
#     global alert_playing
#     if not alert_playing:
#         alert_playing = True
#         try:
#             alert_sound.play()
#         except Exception as e:
#             print(f"[Sound Error] {e}")
#         # Reset alert flag after 2 seconds
#         threading.Timer(2.0, reset_alert_flag).start()

# def compute_EAR(landmarks, eye_indices, w, h):
#     points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
#     A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
#     B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
#     C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
#     ear = (A + B) / (2.0 * C)
#     return ear

# # Start webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     h, w, _ = frame.shape
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     result = face_mesh.process(rgb)

#     if result.multi_face_landmarks:
#         no_face_start_time = None  # Reset absence timer

#         for face_landmarks in result.multi_face_landmarks:
#             # EAR calculation
#             left_ear = compute_EAR(face_landmarks.landmark, LEFT_EYE, w, h)
#             right_ear = compute_EAR(face_landmarks.landmark, RIGHT_EYE, w, h)
#             avg_ear = (left_ear + right_ear) / 2.0

#             # Mouth distance for yawn detection
#             upper_lip_y = face_landmarks.landmark[UPPER_LIP].y
#             lower_lip_y = face_landmarks.landmark[LOWER_LIP].y
#             mouth_open = abs(upper_lip_y - lower_lip_y)

#             # Drowsiness detection
#             if avg_ear < EAR_THRESHOLD:
#                 counter += 1
#                 if counter >= CONSEC_FRAMES:
#                     cv2.putText(frame, "DROWSY!", (30, 100), font, 1.5, (0, 0, 255), 3)
#                     play_alert()
#             else:
#                 counter = 0
#                 cv2.putText(frame, "Awake", (30, 100), font, 1.5, (0, 255, 0), 3)

#             # Yawning detection
#             if mouth_open > MOUTH_OPEN_THRESHOLD:
#                 cv2.putText(frame, "Yawning!", (30, 150), font, 1.5, (0, 0, 255), 3)
#                 play_alert()

#             # 🧠 FACE BOUNDING BOX
#             x_list = [lm.x for lm in face_landmarks.landmark]
#             y_list = [lm.y for lm in face_landmarks.landmark]
#             min_x = int(min(x_list) * w)
#             min_y = int(min(y_list) * h)
#             max_x = int(max(x_list) * w)
#             max_y = int(max(y_list) * h)
#             cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

#     else:
#         # Face not detected
#         if no_face_start_time is None:
#             no_face_start_time = time.time()
#         else:
#             elapsed = time.time() - no_face_start_time
#             if elapsed >= FACE_ABSENCE_THRESHOLD:
#                 cv2.putText(frame, "FACE NOT DETECTED!", (30, 200), font, 1.5, (0, 0, 255), 3)
#                 play_alert()

#     # 🔢 FPS COUNTER
#     curr_time = time.time()
#     fps = 1 / (curr_time - prev_time)
#     prev_time = curr_time
#     cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), font, 0.7, (255, 255, 0), 2)

#     cv2.imshow("Drowsiness + Yawning Detection", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()







