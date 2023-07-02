import cv2
import mediapipe as mp
import numpy as np

def detect_hand_landmarks(frame, landmarks):
    # Store the landmarks' coordinates
    landmark_points = []
    
    # Draw the hand landmarks and store the coordinates
    for landmark in landmarks.landmark:
        x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
        landmark_points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    
    # Connect the landmarks with lines
    if len(landmark_points) > 0:
        cv2.polylines(frame, [np.array(landmark_points, dtype=np.int32)], False, (0, 255, 0), 2)

def main():
    # Initialize VideoCapture
    cap = cv2.VideoCapture(0)

    # Create a hand detector
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    while True:
        # Read frame from the webcam
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with the hand detector
        results = hands.process(frame_rgb)

        # Check if hand landmarks are detected
        if results.multi_hand_landmarks:
            # Draw the hand landmarks on the frame
            for hand_landmarks in results.multi_hand_landmarks:
                detect_hand_landmarks(frame, hand_landmarks)

        # Show the frame with hand landmarks
        cv2.imshow("Hand Recognition", frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture and close the windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
