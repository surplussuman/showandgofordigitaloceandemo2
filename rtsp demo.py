import cv2
import pygame
from pygame.locals import *
import threading

# Load the pre-trained face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize Pygame
pygame.init()

# Create a Pygame window
screen = pygame.display.set_mode((640, 480))


# Initialize the video capture from the RTSP URL
rtsp_url = 'rtsp://admin:admin@123@172.16.21.100:554/cam/realmonitor?channel=5&subtype=0&unicast=true&proto=Onvif'
cap = cv2.VideoCapture(rtsp_url)


# Get the frame width and height
#frame_width = int(cap.get(3))
#frame_height = int(cap.get(4))

# Create a portrait-oriented Pygame window
#screen = pygame.display.set_mode((frame_height, frame_width))


def capture_video():
    while True:
        # Read a frame from the video stream
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640,480))
        # Perform image enhancement and face detection here
        # ...
         # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

        for (x, y, w, h) in faces:
            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Create a thread for video capture
video_thread = threading.Thread(target=capture_video)
video_thread.start()

# Initialize Pygame
pygame.init()

# Create a Pygame window
screen = pygame.display.set_mode((640, 480))

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            video_thread.join()  # Wait for the video thread to finish
            exit()