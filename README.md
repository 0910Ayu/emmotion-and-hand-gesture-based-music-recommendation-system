# Emmotion-and-Hand-Gesture-Based-Music-Recommendation-System
This hybrid music recommendation system combines hand gestures and facial emotion detection for user interaction. It uses MediaPipe with TensorFlow for gesture recognition and a FER algorithm for facial analysis. The system prioritizes gestures, followed by emotions, for selecting music. It identifies Navarasa (nine emotions) using computer vision techniques and maps them to Melakarta ragas for personalized recommendations. Tools used include Python, scikit-learn, and OpenCV.


 This project proposes an AI-based music recommendation system that uses real-time hand gestures and facial emotion recognition via webcam to suggest personalized songs. By combining gesture and emotion detection with machine learning, the system maps emotional states to suitable music (e.g., ragas) to enhance the user's mood and listening experience. Deliverables include a working system and detailed documentation.

<img src="https://github.com/user-attachments/assets/eb85b12f-11b8-4a60-9812-d10cb3b8ef78" alt="flowchart" width="300"/>

<H3>Gesture-Based Expressions:</H3>
Okay – Indicates all is well; plays feel-good or classic songs.
Thumbs Up – Signals agreement or positivity; plays confidence-boosting tracks.
Thumbs Down – Shows disapproval; plays mood-matching songs.
Stop – Used to pause or stop; triggers calm or pause-worthy music.
Rock – Common in rock culture; cues energetic rock music.
Call Me – Suggests a good vibe; plays chill or friendly mood songs.
Live Long – Reflects devotion; plays divine or spiritual music.
Fist – Expresses anger or intensity; plays aggressive or high-energy songs.

<img src="https://github.com/user-attachments/assets/730cd12a-efc7-4238-8834-e887e8b1a38d" alt="flowchart" width="350"/>

<H3> Emotion based expressions: </H3>
 To get familiar
 with the proposed system the user can show the
 available expressions . The
 system will work only on these seven emotion
 based expressions.
 
 <img src="https://github.com/user-attachments/assets/ac783922-d866-476b-a592-90ddcea13c61" alt="flowchart" width="350"/>


Functional testing involves verifying whether the system accurately detects facial emotions and hand gestures, and generates appropriate music recommendations based on these inputs. Accuracy testing evaluates the precision of emotion detection and hand gesture recognition algorithms by comparing system outputs with ground truth data. The hand gesture recognition model achieves 95% accuracy, the emotion recognition model has an accuracy of <H2>82%</H2>, and the overall system accuracy is 88%. Usability testing focuses on assessing the system's ease of use, user interface, and overall experience, with feedback collected from test participants. Performance testing evaluates the system’s speed, responsiveness, and scalability under various conditions, including its ability to process emotion and gesture inputs quickly and handle multiple users simultaneously without significant performance degradation.


