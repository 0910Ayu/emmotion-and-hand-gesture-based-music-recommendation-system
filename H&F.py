import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import logging
logging.disable(logging.WARNING)
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import model_from_json
import time
import random
import pygame
import streamlit as st

global max_gesture 

def facial_emotion_detection():
    model = model_from_json(open("static\Final_jsonmodel.json", "r").read())
    model.load_weights('static\Final.h5')

    face_haar_cascade = cv2.CascadeClassifier('static\haarcascade_frontalface_default.xml')
    cap=cv2.VideoCapture(0)

    emotion_list = []
    cv2.namedWindow('Facial Emotion Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Facial Emotion Detection', 640, 480)  # Adjust window size

    while len(emotion_list) < 30:
        res, frame = cap.read()

        height, width , channel = frame.shape
        sub_img = frame[0:int(height/6), 0:int(width)]
        black_rect = np.ones(sub_img.shape, dtype=np.uint8)*0
        res = cv2.addWeighted(sub_img, 0.77, black_rect, 0.23, 0)

        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_haar_cascade.detectMultiScale(gray_image)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            cv2.rectangle(frame, pt1=(x, y), pt2=(x+w, y+h), color=(255, 0, 0), thickness=2)
            roi_gray = gray_image[y-5:y+h+5, x-5:x+w+5]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            image_pixels = img_to_array(roi_gray)
            image_pixels = np.expand_dims(image_pixels, axis=0)
            image_pixels /= 255
            predictions = model.predict(image_pixels)
            max_index = np.argmax(predictions[0])
            emotion_detection = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            emotion_prediction = emotion_detection[max_index]
            emotion_list.append(emotion_prediction)

            FONT = cv2.FONT_HERSHEY_SIMPLEX
            FONT_SCALE = 0.7
            FONT_THICKNESS = 2
            lable_color = (10, 10, 255)
            lable_violation = 'Sentiment: {}'.format(emotion_prediction)

            # Apply horizontal flip to the frame (mirror view)
            frame_flipped = cv2.flip(frame, 1)

            # Calculate text position based on the original frame size
            violation_text_dimension = cv2.getTextSize(lable_violation, FONT, FONT_SCALE, FONT_THICKNESS)[0]
            violation_x_axis = int((width - violation_text_dimension[0]) - 10)
            cv2.putText(frame_flipped, lable_violation, (violation_x_axis, int(height/6) - 10), FONT, FONT_SCALE, lable_color, FONT_THICKNESS)

            # Display the mirrored view
            cv2.imshow('Facial Emotion Detection', frame_flipped)
            cv2.setWindowProperty('Facial Emotion Detection', cv2.WND_PROP_TOPMOST, 1)  # Make window stay on top
            cv2.waitKey(60)  # Capture for 60 milliseconds

    cap.release()
    cv2.destroyAllWindows()

    # Count occurrences of each emotion
    emotion_counts = {emotion: emotion_list.count(emotion) for emotion in set(emotion_list)}

    # Find the emotion with the maximum count
    max_gesture = max(emotion_counts, key=emotion_counts.get)
    print(emotion_list)
    if max_gesture != 'neutral':
        print("Most detected emotion:", max_gesture)
        play_random_song(max_gesture)
    else:
        # Check the second most detected emotion
        del emotion_counts[max_gesture]  # Remove 'neutral' from counts
        second_max_gesture = max(emotion_counts, key=emotion_counts.get)

        # Check if the count of the second max emotion is greater than or equal to 9
        if emotion_counts[second_max_gesture] >= 9:
            print("Second most detected emotion:", second_max_gesture)
            play_random_song(second_max_gesture)
            second_max_gesture = max_gesture
        else:
            print("Try Hand Gesture")

gesture_emotion_map = {"peace ":"sad","live long":"sad","thumbs down ": "disgust","stop ":"fear","rock ": "surprise","call me":"surprise","fist ":"angry","smile ":"happy","okay ":"happy","thumbs up ":"happy"}
        


def hand_gesture_detection():
    # initialize mediapipe
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mpDraw = mp.solutions.drawing_utils

    # Load the gesture recognizer model
    model = load_model('mp_hand_gesture')

    # Load class names
    with open('gesture.names', 'r') as f:
        classNames = f.read().split('\n')

    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    

    gesture_history = []  # Variable to store detected gestures 

    last_detection_time = time.time()

    cv2.namedWindow('Hand Gesture Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Hand Gesture Detection', 640, 480)  # Adjust window size 

    while True:
        # Read each frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        x, y, c = frame.shape

        # Flip the frame vertically
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get hand landmark prediction
        result = hands.process(framergb)

        className = ''

        # post-process the result
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)
                    landmarks.append([lmx, lmy])

                # Drawing landmarks on frames
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

                # Predict gesture
                prediction = model.predict([landmarks])
                classID = np.argmax(prediction)
                className = classNames[classID]

                # Store detected gesture in variable
                gesture_history.append(className)

                last_detection_time = time.time()

        # show the prediction on the frame
        cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0,0,255), 2, cv2.LINE_AA)

        # Show the webcam view
        cv2.imshow("Hand Gesture Detection", frame)
        cv2.setWindowProperty('Hand Gesture Detection', cv2.WND_PROP_TOPMOST, 1)  # Make window stay on top

        # Automatically close after storing the gesture
        if time.time() - last_detection_time >= 5:
            break
        if len(gesture_history) >= 30:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release the webcam and destroy all active windows
    cap.release()
    cv2.destroyAllWindows()

    # Access gesture_history variable to see detected gestures
    print("Detected Gestures:", gesture_history)

    # Find the most common gesture in gesture_history (All Brain cells died here!!)
    max_gesture = max(set(gesture_history), key=gesture_history.count)
    
    a = gesture_emotion_map[max_gesture]
    print("Folder :- ",a)
    print("Max Gesture:", max_gesture)
    play_random_song(a)

def play_random_song(folder_path):
    # Get list of files in the folder
    files = os.listdir(folder_path)
    # Filter out non-audio files (you might need to adjust this based on your file types)
    audio_files = [file for file in files if file.endswith(('.mp3', '.wav'))]
    
    if not audio_files:
        print("No audio files found in the folder.")
        return
    
    # Choose a random audio file
    random_audio_file = random.choice(audio_files)
    audio_path = os.path.join(folder_path, random_audio_file)
    
    try:
        # Initialize pygame
        pygame.mixer.init()
        
        # Print the name of the song
        print(f"Playing: {random_audio_file}")
        
        # Load the audio file as a Sound object
        sound = pygame.mixer.Sound(audio_path)
        
        # Play the audio
        sound.play()
        
        # Add a delay equal to the length of the audio
        pygame.time.wait(int(sound.get_length() * 1000))
        
    except pygame.error as e:
        print(f"Error occurred: {e}")   
    






def main():
    choice = input("Choose detection type \n 1: Facial Emotion Detection \n 2: Hand Gesture Detection\nYour choice: ")
    if choice == '1':
        facial_emotion_detection()
    elif choice == '2':
        hand_gesture_detection()
    else:
        print("Invalid choice")
    
if __name__ == "__main__":
    main()
