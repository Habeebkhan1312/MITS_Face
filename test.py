import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('unique_face_expression_model_.h5')

# Set image dimensions and class labels
img_height, img_width = 48, 48
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy','Neutral','Sad','Surprise']  # Adjust if your dataset classes differ

# Start video capture (0 is typically the default webcam)
cap = cv2.VideoCapture(0)

# Define a face detector using OpenCV's pre-trained Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection and model input
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Extract the face ROI and resize it to match the model's input size
        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (img_height, img_width))
        face_resized = face_resized.astype('float32') / 255.0  # Normalize the image
        face_resized = np.expand_dims(face_resized, axis=-1)  # Add channel dimension
        face_resized = np.expand_dims(face_resized, axis=0)  # Add batch dimension

        # Predict the expression
        prediction = model.predict(face_resized)
        predicted_class = np.argmax(prediction)
        predicted_label = class_labels[predicted_class]

        # Display the label on the frame
        cv2.putText(frame, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Display the frame
    cv2.imshow('Real-Time Face Expression Recognition', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
