import cv2
from deepface import DeepFace
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from yellowbrick.classifier import ConfusionMatrix

# Initialize lists to store the true and predicted emotion labels
true_emotions = []
pred_emotions = []

# Define function to perform emotion analysis on a face ROI
def analyze_emotion(face_roi):
    results = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
    dominant_emotion = max(results[0]['emotion'].items(), key=lambda x: x[1])[0]
    return dominant_emotion

# Define function to generate the confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(10, 8))
    cm_vis = ConfusionMatrix(None, classes=classes)
    cm_vis.score(y_true, y_pred)
    cm_vis.poof(outpath=None, ax=ax)
    plt.title('Confusion Matrix')
    plt.show()

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained emotion model from DeepFace
model = DeepFace.build_model('Emotion')

# Start capturing real-time video
cap = cv2.VideoCapture(0)

while True:
    # Capture frame from video
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale frame to RGB format
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Extract the face ROI (Region of Interest)
        face_roi = rgb_frame[y:y + h, x:x + w]

        # Resize face ROI to 48x48 pixels (input size for the emotion model)
        face_roi = cv2.resize(face_roi, (48, 48))

        # Perform emotion analysis on the face ROI
        pred_emotion = analyze_emotion(face_roi)

        # Display the predicted emotion label on the frame
        cv2.putText(frame, pred_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Wait for user input to label the true emotion
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            true_emotions.append('Angry')
        elif key == ord('2'):
            true_emotions.append('Disgust')
        elif key == ord('3'):
            true_emotions.append('Fear')
        elif key == ord('4'):
            true_emotions.append('Happy')
        elif key == ord('5'):
            true_emotions.append('Sad')
        elif key == ord('6'):
            true_emotions.append('Surprise')

        # Append the predicted emotion to the list
        pred_emotions.append(pred_emotion)

    # Display the resulting frame
    cv2.imshow('Real-time Emotion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()

# Generate the confusion matrix
classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']
plot_confusion_matrix(true_emotions, pred_emotions, classes)

