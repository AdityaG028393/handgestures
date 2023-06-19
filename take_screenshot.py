import cv2
import mediapipe as mp
import pyautogui

# Initialize the necessary components
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Set up screen capture dimensions
screen_width, screen_height = pyautogui.size()

# Set up the hand detection model
with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7) as hands:
    # Set up the webcam for video capture
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Error reading frame from webcam.")
            break

        # Flip the image horizontally for natural movement mirroring
        image = cv2.flip(image, 1)

        # Convert the image to RGB for processing by Mediapipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # Check for detected hands and corresponding landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get the landmarks for thumb and index finger
                thumb_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_finger_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Calculate the distance between thumb and index finger
                distance = abs(thumb_landmark.x - index_finger_landmark.x)

                # Check if the distance is below a certain threshold for thumbs-up gesture
                if distance < 0.05:
                    # Capture a screenshot using PyAutoGUI
                    screenshot = pyautogui.screenshot()
                    screenshot.save('thumbs_up_screenshot.png')
                    print("Thumbs-up gesture detected. Screenshot captured.")

                # Check if the distance is above a certain threshold for thumbs-down gesture
                if distance > 0.2:
                    # Capture a screenshot using PyAutoGUI
                    screenshot = pyautogui.screenshot()
                    screenshot.save('thumbs_down_screenshot.png')
                    print("Thumbs-down gesture detected. Screenshot captured.")

        # Draw hand landmarks on the image
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Show the image with landmarks
        cv2.imshow('Hand Gestures', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
