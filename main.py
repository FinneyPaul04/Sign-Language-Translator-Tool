# Import necessary libraries
import numpy as np
import os
import string
import mediapipe as mp  # Corrected import
import cv2
from my_functions import *
import keyboard
from tensorflow.keras.models import load_model
import language_tool_python

def main():
    try:
        # Set the path to the data directory
        PATH = os.path.join('data')
        if not os.path.exists(PATH):
            raise FileNotFoundError(f"Data directory not found at {PATH}")

        # Get action labels
        actions = np.array([f for f in os.listdir(PATH) if not f.startswith('.')])  # Skip hidden files
        
        # Load the trained model
        try:
            model = load_model('my_model.h5')  # Assuming .h5 format
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

        # Initialize language tool
        tool = language_tool_python.LanguageTool('en-UK')

        # Initialize variables
        sentence = []
        keypoints = []
        last_prediction = None
        grammar_result = None

        # Camera setup
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot access camera.")

        # MediaPipe setup
        holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75
        )

        # Main loop
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            try:
                results = image_process(frame, holistic)
                draw_landmarks(frame, results)
                current_kp = keypoint_extraction(results)
                keypoints.append(current_kp)
            except Exception as e:
                print(f"Frame processing error: {e}")
                continue

            # Prediction logic
            if len(keypoints) == 10:
                try:
                    prediction = model.predict(np.expand_dims(keypoints, axis=0))
                    keypoints = []  # Reset buffer
                    
                    if np.amax(prediction) > 0.9:
                        current_action = actions[np.argmax(prediction)]
                        if last_prediction != current_action:
                            sentence.append(current_action)
                            last_prediction = current_action
                except Exception as e:
                    print(f"Prediction error: {e}")

            # Sentence management
            if len(sentence) > 7:
                sentence = sentence[-7:]
                
            if keyboard.is_pressed(' '):  # Reset
                sentence, keypoints, grammar_result = [], [], None
                
            if sentence:
                sentence[0] = sentence[0].capitalize()
                if len(sentence) >= 2:
                    process_sentence(sentence, actions)

            # Grammar check
            if keyboard.is_pressed('enter') and sentence:
                try:
                    grammar_result = tool.correct(' '.join(sentence))
                except Exception as e:
                    print(f"Grammar check failed: {e}")

            # Display
            display_text = grammar_result if grammar_result else ' '.join(sentence)
            if display_text:
                put_centered_text(frame, display_text)

            cv2.imshow('Sign Language Translator', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        if 'holistic' in locals():
            holistic.close()
        tool.close()

def process_sentence(sentence, actions):
    """Process sentence formatting rules"""
    if (len(sentence) >= 2 and 
        sentence[-1] in string.ascii_letters and
        (sentence[-2] in string.ascii_letters or 
         sentence[-2].lower() not in actions)):
        sentence[-1] = sentence[-2] + sentence[-1]
        sentence.pop(-2)
        sentence[-1] = sentence[-1].capitalize()

def put_centered_text(img, text):
    """Display centered text on image"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    thickness = 2
    color = (255, 255, 255)
    
    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
    text_x = (img.shape[1] - text_size[0]) // 2
    cv2.putText(img, text, (text_x, 470), font, scale, color, thickness, cv2.LINE_AA)

if __name__ == "__main__":
    main()