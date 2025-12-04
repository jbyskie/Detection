import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)

frame_countdown = 60  # countdown for calibration


# ---------------------------------------------------------
# COUNT FINGERS (WITH CORRECT LEFT/RIGHT THUMB LOGIC)
# ---------------------------------------------------------
def count_fingers(hand_landmarks, handedness_label):
    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
    fingers_up = 0
    lm = hand_landmarks.landmark

    # Other 4 fingers
    for tip in finger_tips:
        if lm[tip].y < lm[tip - 2].y:   # tip above PIP joint
            fingers_up += 1

    # Thumb (different for L & R)
    thumb_tip = lm[4].x
    thumb_ip = lm[3].x

    if handedness_label == "Right":
        if thumb_tip < thumb_ip:  # right thumb opens outward (left direction)
            fingers_up += 1
    else:  # LEFT HAND
        if thumb_tip > thumb_ip:  # left thumb opens outward (right direction)
            fingers_up += 1

    return fingers_up


# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Show countdown
    cv2.putText(frame, f"Countdown: {frame_countdown}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if frame_countdown > 0:
        frame_countdown -= 1

    else:
        results = hands.process(rgb)

        if results.multi_hand_landmarks and results.multi_handedness:

            for handLM, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

                label = handedness.classification[0].label  # "Left" or "Right"

                # Draw landmarks
                mp_draw.draw_landmarks(frame, handLM, mp_hands.HAND_CONNECTIONS)

                # Bounding Box
                h, w, _ = frame.shape
                x_vals = [lm.x * w for lm in handLM.landmark]
                y_vals = [lm.y * h for lm in handLM.landmark]

                x_min, x_max = int(min(x_vals)), int(max(x_vals))
                y_min, y_max = int(min(y_vals)), int(max(y_vals))

                cv2.rectangle(frame, (x_min - 10, y_min - 10),
                              (x_max + 10, y_max + 10),
                              (0, 255, 255), 2)

                # COUNT FINGERS with correct left/right thumb
                fingers = count_fingers(handLM, label)

                cv2.putText(frame, f"{label}: {fingers}", (x_min, y_min - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Finger Counter", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("r"):
        frame_countdown = 60
        print("Restarting detection...")

cap.release()
cv2.destroyAllWindows()
