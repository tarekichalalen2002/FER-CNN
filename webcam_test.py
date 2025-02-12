import cv2
import torch
import joblib
import numpy as np
from FERCNN import FERCNN  # Ensure this is correctly imported
from torchvision import transforms
from PIL import Image  
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load the trained model
model_filename = "FER-CNN.pkl"
model = joblib.load(model_filename)
model.eval()  # Set to evaluation mode
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Emotion labels (adjust based on your dataset)
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
# Start capturing video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = gray_frame[y:y + h, x:x + w]  # Extract face ROI

        # Preprocess the face
        face_pil = cv2.resize(face, (48, 48))  # Resize to match model input
        face_pil = Image.fromarray(face)
        face_tensor = transform(face_pil).unsqueeze(0).to(device)

        # Predict emotion
        with torch.no_grad():
            output = model(face_tensor)
            _, predicted = torch.max(output, 1)
            emotion = emotion_labels[predicted.item()]

        # Draw rectangle around face and label it
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show video feed
    cv2.imshow("Facial Expression Recognition", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
