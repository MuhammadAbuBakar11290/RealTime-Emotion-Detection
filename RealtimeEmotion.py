import cv2
import os
import sys
from keras.models import load_model
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.stdout.reconfigure(encoding='utf-8')
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Load the model
# Load the trained model

try:
    model = load_model('emotiondetector.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()


# Initialize the face detector
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Define labels for emotions
labels = {
    0: 'angry', 
    1: 'disgust', 
    2: 'fear', 
    3: 'happy', 
    4: 'neutral', 
    5: 'sad', 
    6: 'surprise'
}

# Function to preprocess the image for prediction
def preprocess_image(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)  # Reshape for model input
    return feature / 255.0  # Normalize

# Start video capture
webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        face_image = gray_frame[y:y+h, x:x+w]
        face_image = cv2.resize(face_image, (48, 48))
        img_features = preprocess_image(face_image)
        
        try:
            pred = model.predict(img_features)
            prediction_label = labels[np.argmax(pred)]
            
            # Draw rectangle around face and display emotion
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, prediction_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        except UnicodeEncodeError as e:
            print(f"UnicodeEncodeError: {e}")
            # Handle the error appropriately, maybe skip this frame or log the issue
    
    # Show the frame with detected faces and emotions
    cv2.imshow("Emotion Detection", frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close windows
webcam.release()
cv2.destroyAllWindows()