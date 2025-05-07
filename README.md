# Emmotion-and-Hand-Gesture-Based-Music-Recommendation-System
Music recommendation that combines hand gestures and facial emotions for user interaction. Current
 research in music recommendation focuses on either hand gesture-based controllers or emotion-based
 players, but not both. The proposed hybrid method utilizes a facial expression recognizer (FER) algorithm
 for emotion detection and the MediaPipe framework with TensorFlow for hand detection and gesture
 recognition. Music selection is based on the most recent gesture and emotion detected, with priority given
 to hand gestures followed by facial emotions. The accuracy of this approach is compared with existing
 methods in music recommendation.
 The methodology involves identification of Navarasa (nine emotions) from camera input. Computer vision
 techniques are employed to analyze facial expressions and detect emotions such as Hasya, Karuna,
 Shringara, etc. These identified Navarasas are then mapped to corresponding emotional tags associated
 with Melakarta ragas. The recommendation system is developed to map the detected Navarasa from
 camera input to emotional tags and provide personalized music recommendations based on the Melakarta
 ragas associated with those emotions. The simulation tools used include programming languages like
 Python, machine learning libraries like scikitlearn and computer vision frameworks like OpenCV .

 This project proposes an AI-based music recommendation system that uses real-time hand gestures and facial emotion recognition via webcam to suggest personalized songs. By combining gesture and emotion detection with machine learning, the system maps emotional states to suitable music (e.g., ragas) to enhance the user's mood and listening experience. Deliverables include a working system and detailed documentation.


