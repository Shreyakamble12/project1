import cv2
import numpy as np

# Importing Models and setting mean values
face_model_txt = "opencv_face_detector.pbtxt"
face_model_bin = "opencv_face_detector_uint8.pb"
age_model_txt = "age_deploy.prototxt"
age_model_bin = "age_net.caffemodel"
gender_model_txt = "gender_deploy.prototxt"
gender_model_bin = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# Load the models
face_net = cv2.dnn.readNet(face_model_bin, face_model_txt)
age_net = cv2.dnn.readNet(age_model_bin, age_model_txt)
gender_net = cv2.dnn.readNet(gender_model_bin, gender_model_txt)

# Capture video from the camera
cap = cv2.VideoCapture(0)

# Motion detection initialization
_, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (25, 25), 0)
previous_frame = gray
motion_threshold = 30

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for motion detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (25, 25), 0)

    # Calculate frame difference for motion detection
    frame_difference = cv2.absdiff(previous_frame, gray)
    _, threshold = cv2.threshold(frame_difference, motion_threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around moving objects
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Face detection for age and gender prediction
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    face_net.setInput(blob)
    detections = face_net.forward()
    face_boxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detections[0, 0, i, 3] * frame.shape[1])
            y1 = int(detections[0, 0, i, 4] * frame.shape[0])
            x2 = int(detections[0, 0, i, 5] * frame.shape[1])
            y2 = int(detections[0, 0, i, 6] * frame.shape[0])
            face_boxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Age and gender prediction
    for face_box in face_boxes:
        face_region = frame[max(0, face_box[1] - 15): min(face_box[3] + 15, frame.shape[0] - 1),
                            max(0, face_box[0] - 15): min(face_box[2] + 15, frame.shape[1] - 1)]

        # Prepare blob for age and gender prediction
        blob = cv2.dnn.blobFromImage(face_region, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Predict gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]

        # Predict age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]

        # Display age and gender
        cv2.putText(frame, f'{gender}, {age}', (face_box[0], face_box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Age, Gender, and Motion Detection', frame)

    # Update previous frame for motion detection
    previous_frame = gray

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
    
