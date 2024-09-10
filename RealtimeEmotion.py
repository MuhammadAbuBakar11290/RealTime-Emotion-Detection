import cv2
import os
import sys
from keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.stdout.reconfigure(encoding='utf-8')
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Load the model
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

# Initialize counters for each emotion
emotion_counters = {emotion: 0 for emotion in labels.values()}
total_frames = 0

# CSV and chart file paths
csv_filename = 'F:/Projects ML/Emotion Detection/emotion_report.csv'
bar_chart_filename = 'F:/Projects ML/Emotion Detection/emotion_bar_chart.png'
pie_chart_filename = 'F:/Projects ML/Emotion Detection/emotion_pie_chart.png'

# Function to preprocess the image for prediction
def preprocess_image(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)  # Reshape for model input
    return feature / 255.0  # Normalize

# Function to update the CSV file and plot charts
def update_report():
    if total_frames > 0:
        # Calculate percentages
        emotion_percentages = {emotion: (count / total_frames) * 100 for emotion, count in emotion_counters.items()}
    else:
        emotion_percentages = {emotion: 0 for emotion in labels.values()}
    
    # Save to CSV
    df = pd.DataFrame(list(emotion_percentages.items()), columns=['Emotion', 'Percentage'])
    df['Percentage'] = df['Percentage'].apply(lambda x: f'{x:.2f}%')
    
    try:
        df.to_csv(csv_filename, index=False)
        print(f"CSV file saved as {csv_filename}")

        # Bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(df['Emotion'], [float(p.strip('%')) for p in df['Percentage']], color='skyblue')
        plt.xlabel('Emotion')
        plt.ylabel('Percentage')
        plt.title('Percentage of Detected Emotions')
        plt.tight_layout()
        plt.savefig(bar_chart_filename)
        plt.close()
        print(f"Bar chart saved as {bar_chart_filename}")

        # Pie chart
        plt.figure(figsize=(8, 8))
        plt.pie([float(p.strip('%')) for p in df['Percentage']],
                labels=df['Emotion'], autopct='%1.1f%%', colors=['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'cyan'], startangle=140)
        plt.title('Emotion Distribution')
        plt.axis('equal')
        plt.savefig(pie_chart_filename)
        plt.close()
        print(f"Pie chart saved as {pie_chart_filename}")
    
    except Exception as e:
        print(f"Error occurred while saving CSV or plotting: {e}")

# Start video capture
webcam = cv2.VideoCapture(0)

try:
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
                
                # Update emotion counters
                emotion_counters[prediction_label] += 1
                total_frames += 1
                
                # Draw rectangle around face and display emotion
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, prediction_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            except Exception as e:
                print(f"Error during prediction: {e}")
        
        # Show the frame with detected faces and emotions
        cv2.imshow("Emotion Detection", frame)
        
        # Update report after every frame
        update_report()
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error occurred: {e}")

finally:
    # Release resources and close windows
    webcam.release()
    cv2.destroyAllWindows()
