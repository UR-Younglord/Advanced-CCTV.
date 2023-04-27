import cv2
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model('crime_detection_model.h5')

# Define the classes
classes = ['burglary', 'infiltration', 'theft', 'unauthorized_access', 'violence']

# Define the video capture
cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object for saving the video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

# Analyze the frames
while True:
    # Capture the frame
    ret, frame = cap.read()

    # Resize the frame to match the input size of the model
    resized_frame = cv2.resize(frame, (224, 224))

    # Preprocess the frame
    preprocessed_frame = tf.keras.applications.mobilenet_v2.preprocess_input(resized_frame)

    # Predict the class of the frame
    prediction = model.predict(tf.expand_dims(preprocessed_frame, axis=0))[0]
    predicted_class = classes[prediction.argmax()]

    # Draw the predicted class label on the frame
    cv2.putText(frame, predicted_class, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Write the frame to the video file
    out.write(frame)

    # Display the frame
    cv2.imshow('frame', frame)

    # Check if the predicted class is a crime
    if predicted_class in ['burglary', 'theft', 'unauthorized_access', 'violence']:
     # Send an alert to the police
    # Code for sending an alert to the nearest police station goes here

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture, writer and destroy the windows
cap.release()
out.release()
cv2.destroyAllWindows()