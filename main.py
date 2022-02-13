import cv2
import mediapipe as mp
import numpy as np
import argparse
from pythonosc import udp_client

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


# Draw Calculate Fingers Angles

joint_list = [[8, 7, 6], [12, 11, 10], [16, 15, 14], [20, 19, 18], [4, 3, 2]]
joint_list[4]

def draw_finger_angles(image, results, joint_list):
    # Loop through hands
    for hand in results.multi_hand_landmarks:
        # Loop through joint sets
        for joint in joint_list:
            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y])  # First coord
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y])  # Second coord
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y])  # Third coord

            radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
            angle = np.abs(radians * 180.0 / np.pi)

            if angle > 180.0:
                angle = 360 - angle

            cv2.putText(image, str(round(angle, 2)), tuple(np.multiply(b, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (15, 15, 15), 1, cv2.LINE_AA)
    return image


# Save finger angles data to join_angle

joint_angle = [180,180,180,180,180]
joint_angle[0]

def finger_angles(results, joint_list, joint_angle):

    for hand in results.multi_hand_landmarks:
        for joint in joint_list:
            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y])  # First coord
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y])  # Second coord
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y])  # Third coord

            radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
            angle = np.abs(radians * 180.0 / np.pi)

            if angle > 180.0:
                angle = 360 - angle
            if joint == [8, 7, 6]:
                joint_angle[0] = angle
            if joint == [12, 11, 10]:
                joint_angle[1] = angle
            if joint == [16, 15, 14]:
                joint_angle[2] = angle
            if joint == [20, 19, 18]:
                joint_angle[3] = angle
            if joint == [4, 3, 2]:
                joint_angle[4] = angle

    return joint_angle


# Detect Hands and Draw Data

cap = cv2.VideoCapture(1)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()

        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Flip on horizontal
        image = cv2.flip(image, 1)

        # Set flag
        image.flags.writeable = False

        # Detections
        results = hands.process(image)

        # Set flag to true
        image.flags.writeable = True

        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Detections
        print(results)

        # Rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(50, 50, 50), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(250, 50, 50), thickness=2, circle_radius=2),
                                          )


            # Draw angles to image from joint list
            draw_finger_angles(image, results, joint_list)
            finger_angles(results, joint_list, joint_angle)
            print(joint_angle)

        # Save our image
        # cv2.imwrite(os.path.join('Output Images', '{}.jpg'.format(uuid.uuid1())), image)
        cv2.imshow('Hand Tracking', image)

        # Send it via OSC
        if __name__ == "__main__":
            parser = argparse.ArgumentParser()
            parser.add_argument("--ip", default="127.0.0.1", help="The ip of the OSC server")
            parser.add_argument("--port", type=int, default=5005, help="The port the OSC server is listening on")
            args = parser.parse_args()

            # Send joint_angle
            client = udp_client.SimpleUDPClient(args.ip, args.port)
            client.send_message("/handOSC", joint_angle)
            # for x in range(10):
            #     client.send_message("/o", joint_angle)
            #     time.sleep(0)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()