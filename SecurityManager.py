# Defining main function
import cv2
import numpy as np
import pygame
from matplotlib import pyplot as plt

import Constants
import main
from ManageUser import getUserNamePassword
from PasswordUtility import verifyPwd
from QR import getQR


def TouchlessSecurityManager():
    print("Starting Touch-less Security Manager")
    states = ['1. HOME', '2. SCAN QR', '3. HELLO', '4. WELCOME']
    stateInfos = ['Show V to start', 'Show _l..l  to go home', 'Show 10 fingers to re-enter PIN',
                  'Show _l..l  to go home']

    state = states[0]
    stateInfo = stateInfos[0]
    info = ""

    # Initialize the pygame modules and load the image-capture music file.
    pygame.init()

    # Initialize the number of consecutive frames on which we want to check the hand gestures before triggering the events.
    num_of_frames = 40

    previous = "10"

    eq, pin = "", "";

    # Initialize a dictionary to store the counts of the consecutive frames with the hand gestures recognized.
    counter = {'V SIGN': 0, 'SPIDERMAN SIGN': 0, 'PIN': 0}

    # Initialize a variable to store the captured image.
    captured_image = None

    # Initialize the VideoCapture object to read from the webcam.
    camera_video = cv2.VideoCapture(0)
    camera_video.set(Constants.width, Constants.pixelWidth)
    camera_video.set(Constants.height, Constants.pixelHeight)

    # Create named window for resizing purposes.
    cv2.namedWindow('Touch less Security System', cv2.WINDOW_NORMAL)

    # Iterate until the webcam is accessed successfully.
    while camera_video.isOpened():

        # Read a frame.
        ok, frame = camera_video.read()

        # Check if frame is not read properly then continue to the next iteration to read the next frame.
        if not ok:
            continue

        # Flip the frame horizontally for natural (selfie-view) visualization.
        frame = cv2.flip(frame, 1)

        cleanFrame = frame.copy()

        # Get the height and width of the frame of the webcam video.
        frame_height, frame_width, _ = frame.shape

        # print(state)
        main.showText(frame, state, stateInfo, info)

        if state == states[1]:
            frame, eq = getQR(frame)
            if eq != "":
                name, password = getUserNamePassword(eq)
                if name != "":
                    main.playMusic('beep')
                    state = states[2]
                    info = name + ", Enter PIN"
                else:
                    info = ". Sorry, failed to recognize!"

        # Perform Hands landmarks detection on the frame.
        frame, results = main.detectHandsLandmarks(frame, main.hands_videos, display=False)

        fingers_statuses, count = 0, 0

        if pin != "":
            main.showPin(frame, pin)

        # Enter PIN
        if results.multi_hand_landmarks and state == states[2]:
            # Count the number of fingers up of each hand in the frame.
            frame, fingers_statuses, count = main.countFingers(frame, results, draw=False, display=False)

            total = str(count['RIGHT'] + count['LEFT'])

            # print(total, counter['PIN'], previous, pin)

            if total == previous:
                counter['PIN'] += 1
                if counter['PIN'] >= num_of_frames:
                    counter['PIN'] = 0
                    if total == "10":
                        pin = ""
                        stateInfo = stateInfos[3]
                    else:
                        pin += total
                        main.playMusic('pin')
                        stateInfo = stateInfos[2]
                        if len(pin) == 4:
                            if verifyPwd(pin, password):
                                state = states[3]
                                stateInfo = stateInfos[3]
                                pin = ""
                                info = name + ", Authorized!"
                                main.playMusic('beep')
                            else:
                                info = name + ", Incorrect PIN. Try Again."
                                main.playMusic('error')
                                pin = ""
                                stateInfo = stateInfos[3]
            else:
                counter['PIN'] = 0

            previous = total

        # Check if the hands landmarks in the frame are detected.
        if results.multi_hand_landmarks:
            # Count the number of fingers up of each hand in the frame.
            frame, fingers_statuses, count = main.countFingers(frame, results, display=False)

            # Perform the hand gesture recognition on the hands in the frame.
            _, hands_gestures = main.recognizeGestures(frame, fingers_statuses, count, draw=False, display=False)

            # Apply and Remove Image Filter Functionality.
            ####################################################################################################################

            # Check if any hand is making the SPIDERMAN hand gesture in the required number of consecutive frames.
            ####################################################################################################################

            # Check if the gesture of any hand in the frame is SPIDERMAN SIGN.
            if any(hand_gesture == "SPIDERMAN SIGN" for hand_gesture in hands_gestures.values()):

                # Increment the count of consecutive frames with SPIDERMAN hand gesture recognized.
                counter['SPIDERMAN SIGN'] += 1

                # Check if the counter is equal to the required number of consecutive frames.
                if counter['SPIDERMAN SIGN'] == num_of_frames:

                    main.playMusic('beep')

                    # RESET
                    state = states[0]
                    stateInfo = stateInfos[0]
                    captured_image = None
                    info = ""
                    eq = ""
                    name = ""
                    password = ""
                    # Update the counter value to zero.
                    counter['SPIDERMAN SIGN'] = 0
                    pin = ""

            # Otherwise if the gesture of any hand in the frame is not SPIDERMAN SIGN.
            else:

                # Update the counter value to zero. As we are counting the consective frames with SPIDERMAN hand
                # gesture.
                counter['SPIDERMAN SIGN'] = 0

            ####################################################################################################################

        # Image Capture Functionality.
        ########################################################################################################################

        # Check if the hands landmarks are detected and the gesture of any hand in the frame is V SIGN.
        if state == states[0] and results.multi_hand_landmarks and any(
                hand_gesture == "V SIGN" for hand_gesture in hands_gestures.values()):

            # Increment the count of consecutive frames with V hand gesture recognized.
            counter['V SIGN'] += 1

            # Check if the counter is equal to the required number of consecutive frames.
            if counter['V SIGN'] == num_of_frames:
                state = states[1]
                stateInfo = stateInfos[1]

                # Make a border around a copy of the current frame.
                captured_image = cv2.copyMakeBorder(src=cleanFrame, top=10, bottom=10, left=10, right=10,
                                                    borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))

                # Capture an image and store it in the disk.
                cv2.imwrite('Captured_Image.png', captured_image)

                # Display a black image.
                cv2.imshow('Touch less Security System', np.zeros((frame_height, frame_width)))

                # Play the image capture music to indicate the an image is captured and wait for 100 milliseconds.
                main.playMusic('cam')

                # Display the captured image.
                plt.close();
                plt.figure(figsize=[10, 10])
                plt.imshow(frame[:, :, ::-1]);
                plt.title("Captured Image");
                plt.axis('off');

                # Update the counter value to zero.
                counter['V SIGN'] = 0

        # Otherwise if the gesture of any hand in the frame is not V SIGN.
        else:

            # Update the counter value to zero. As we are counting the consective frames with V hand gesture.
            counter['V SIGN'] = 0

        ########################################################################################################################

        # Check if we have captured an image.
        if captured_image is not None:
            # Resize the image to the 1/5th of its current width while keeping the aspect ratio constant.
            captured_image = cv2.resize(captured_image,
                                        (frame_width // 5, int(((frame_width // 5) / frame_width) * frame_height)))

            # Get the new height and width of the image.
            img_height, img_width, _ = captured_image.shape

            # Overlay the resized captured image over the frame by updating its pixel values in the region of interest.
            frame[frame_height - 10 - img_height: frame_height - 10, 10: 10 + img_width] = captured_image

        # Visualize the counted fingers.
        frame = main.annotate(frame, results, fingers_statuses, count, display=False)

        # Display the frame.
        cv2.imshow('Touch less Security System', frame)

        # Wait for 1ms. If a key is pressed, retreive the ASCII code of the key.
        k = cv2.waitKey(1) & 0xFF

        # Check if 'ESC' is pressed and break the loop.
        if k == 27:
            break

    # Release the VideoCapture Object and close the windows.
    camera_video.release()
    cv2.destroyAllWindows()

    print("Stopping Touch-less Security Manager")


# Using the special variable
# __name__
if __name__ == "__main__":
    print("begin")
    TouchlessSecurityManager()
    print("end")
