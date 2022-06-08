# Defining main function
import cv2
import main


def task():
    print("starting task")

    # Initialize the VideoCapture object to read from the webcam.
    camera_video = cv2.VideoCapture(0)
    camera_video.set(3, 1280)
    camera_video.set(4, 960)

    # Create named window for resizing purposes.
    cv2.namedWindow('Counted Fingers Visualization', cv2.WINDOW_FULLSCREEN)

    # Iterate until the webcam is accessed successfully.
    while camera_video.isOpened():

        # Read a frame.
        ok, frame = camera_video.read()

        # Check if frame is not read properly then continue to the next iteration to read the next frame.
        if not ok:
            continue

        # Flip the frame horizontally for natural (selfie-view) visualization.
        frame = cv2.flip(frame, 1)

        # Perform Hands landmarks detection on the frame.
        frame, results = main.detectHandsLandmarks(frame, main.hands_videos, display=False)

        fingers_statuses, count = 0, 0

        # Check if the hands landmarks in the frame are detected.
        if results.multi_hand_landmarks:
            # Count the number of fingers up of each hand in the frame.
            frame, fingers_statuses, count = main.countFingers(frame, results, display=False)

        # Visualize the counted fingers.
        frame = main.annotate(frame, results, fingers_statuses, count, display=False)

        # Display the frame.
        cv2.imshow('Counted Fingers Visualization', frame)

        # Wait for 1ms. If a key is pressed, retreive the ASCII code of the key.
        k = cv2.waitKey(1) & 0xFF

        # Check if 'ESC' is pressed and break the loop.
        if (k == 27):
            break

    # Release the VideoCapture Object and close the windows.
    camera_video.release()
    cv2.destroyAllWindows()

    print("stopping task")


# Using the special variable
# __name__
if __name__ == "__main__":
    print("start")
    task()
    print("end")
