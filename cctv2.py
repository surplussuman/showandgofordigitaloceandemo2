'''import cv2
import numpy as np

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_image(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur for noise reduction
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Apply histogram equalization for contrast enhancement
    equalized_image = cv2.equalizeHist(blurred_image)
    
    return equalized_image

def detect_faces(image):
    # Detect faces using Haar cascade
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5)
    return faces

def non_max_suppression(boxes, overlapThresh):
    # Non-maximum suppression implementation
    if len(boxes) == 0:
        return []
    
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        overlap = (w * h) / area[idxs[:last]]
        
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    
    return boxes[pick]

rtsp_url = 'rtsp://admin:admin@123@172.16.21.101:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif'

# Capture video from the CCTV camera
cap = cv2.VideoCapture(rtsp_url)  # Use 0 for the default camera

while True:
    # Read the frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame
    preprocessed_frame = preprocess_image(frame)
    
    # Detect faces in the preprocessed frame
    faces = detect_faces(preprocessed_frame)
    
    # Apply non-maximum suppression to remove overlapping detections
    faces = non_max_suppression(faces, overlapThresh=0.3)
    
    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Display the frame
    cv2.imshow('Face Detection', frame)
    
    # Check for 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
'''
'''
import cv2
from facenet_pytorch import MTCNN

# Initialize MTCNN
mtcnn = MTCNN()

# RTSP URL
rtsp_url = "rtsp://admin:LYDLKK@192.168.0.169:554/H.264"

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

# Check if the stream is opened successfully
if not cap.isOpened():
    print("Error opening stream")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB (MTCNN expects RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform face detection
    boxes, probs, landmarks = mtcnn.detect(rgb_frame, landmarks=True)

    if boxes is not None:
        for box, landmark in zip(boxes, landmarks):
            # Draw bounding box and landmarks
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            for (x, y) in landmark.astype(int):
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

    # Display the frame
    cv2.imshow('RTSP Stream', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''

'''import cv2
from facenet_pytorch import MTCNN
import threading
import queue

# Initialize MTCNN
mtcnn = MTCNN()

# RTSP URL
rtsp_url = "rtsp://admin:LYDLKK@192.168.0.169:554/H.264"

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

# Check if the stream is opened successfully
if not cap.isOpened():
    print("Error opening stream")
    exit()

# Queue for frames
frame_queue = queue.Queue(maxsize=1)

# Function to read frames and put them into the queue
def read_frames():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Resize the frame to a smaller size
        resized_frame = cv2.resize(frame, (640, 480))
        frame_queue.put(resized_frame)

# Function to perform face detection and display frames
def detect_faces_and_display():
    while cap.isOpened():
        frame = frame_queue.get()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probs, landmarks = mtcnn.detect(rgb_frame, landmarks=True)
        if boxes is not None:
            for box, landmark in zip(boxes, landmarks):
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                for (x, y) in landmark.astype(int):
                    cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
        cv2.imshow('RTSP Stream', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Start the threads
read_thread = threading.Thread(target=read_frames)
detect_thread = threading.Thread(target=detect_faces_and_display)

read_thread.start()
detect_thread.start()

# Wait for the threads to finish
read_thread.join()
detect_thread.join()

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
'''
'''import cv2
from facenet_pytorch import MTCNN
import threading
import queue
import time

# Initialize MTCNN
mtcnn = MTCNN()

# RTSP URL
rtsp_url = "rtsp://admin:LYDLKK@192.168.0.169:554/H.264"
#rtsp_url = 'http://127.0.0.1:5000/video_feed'

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

# Check if the stream is opened successfully
if not cap.isOpened():
    print("Error opening stream")
    exit()

# Set video capture properties
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Queue for frames
frame_queue = queue.Queue(maxsize=1)

# Function to read frames and put them into the queue
def read_frames():
    last_frame_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Resize the frame to a smaller size
        resized_frame = cv2.resize(frame, (640, 480))
        frame_queue.put(resized_frame)
        last_frame_time = time.time()

        # Check for timeout (5 seconds)
        if time.time() - last_frame_time > 5:
            print("Timeout: No frames received for 5 seconds")
            break

# Function to perform face detection and display frames
def detect_faces_and_display():
    while cap.isOpened():
        frame = frame_queue.get()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probs, landmarks = mtcnn.detect(rgb_frame, landmarks=True)
        if boxes is not None:
            for box, landmark in zip(boxes, landmarks):
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                for (x, y) in landmark.astype(int):
                    cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
        cv2.imshow('RTSP Stream', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Start the threads
read_thread = threading.Thread(target=read_frames)
detect_thread = threading.Thread(target=detect_faces_and_display)

read_thread.start()
detect_thread.start()

# Wait for the threads to finish
read_thread.join()
detect_thread.join()

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
'''
import cv2
from facenet_pytorch import MTCNN
import threading
import queue
import time
import logging

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

# Initialize MTCNN
mtcnn = MTCNN()

# RTSP URL
#rtsp_url = "rtsp://admin:LYDLKK@192.168.0.169:554/H.264"
#rtsp_url = 'video.mp4'
# Open the RTSP stream
cap = cv2.VideoCapture('video.mp4')

# Check if the stream is opened successfully
if not cap.isOpened():
    logging.error("Error opening stream")
    exit()

# Set video capture properties
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Queue for frames
frame_queue = queue.Queue(maxsize=1)

# Function to read frames and put them into the queue
def read_frames():
    last_frame_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logging.error("Error reading frame")
            break
        # Resize the frame to a smaller size
        resized_frame = cv2.resize(frame, (640, 480))
        frame_queue.put(resized_frame)
        last_frame_time = time.time()

        # Check for timeout (5 seconds)
        if time.time() - last_frame_time > 5:
            logging.error("Timeout: No frames received for 5 seconds")
            break

# Function to perform face detection and display frames
'''def detect_faces_and_display():
    while cap.isOpened():
        frame = frame_queue.get()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probs, landmarks = mtcnn.detect(rgb_frame, landmarks=True)
        if boxes is not None:
            for box, landmark in zip(boxes, landmarks):
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                for (x, y) in landmark.astype(int):
                    cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
                
        cv2.imshow("Steram", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        '''
def detect_faces_and_display():
    # Create windows outside the loop
    cv2.namedWindow("Mark Attendance In - Press q to Quit")
    cv2.moveWindow("Mark Attendance In - Press q to Quit", 800, 0)
    cv2.namedWindow("Detected Face")
    cv2.moveWindow("Detected Face", 800, 0)


    small_window_width = 300
    small_window_height = 300

    while cap.isOpened():
        frame = frame_queue.get()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probs, landmarks = mtcnn.detect(rgb_frame, landmarks=True)
        if boxes is not None:
            for box, landmark in zip(boxes, landmarks):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                for (x, y) in landmark.astype(int):
                    cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
                rgb_face = frame[y1:y2, x1:x2]
                zoomed_face_region = cv2.resize(rgb_face, (small_window_width, small_window_height))
                cv2.imshow("Detected Face", zoomed_face_region)

        cv2.imshow("Mark Attendance In - Press q to Quit", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break



# Start the threads
read_thread = threading.Thread(target=read_frames)
detect_thread = threading.Thread(target=detect_faces_and_display)

read_thread.start()
detect_thread.start()

# Wait for the threads to finish
read_thread.join()
detect_thread.join()

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
