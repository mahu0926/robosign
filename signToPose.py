# take in an image
# process image through mediapose
# get points from the image

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np

model_path = '/Users/mahathimanda/Desktop/build18/hand_landmarker.task'

# array of landmark names
landmark_names = [
    'wrist',
    'thumb_cmc',
    'thumb_mcp',
    'thumb_ip',
    'thumb_tip',
    'index_finger_mcp',
    'index_finger_pip',
    'index_finger_dip',
    'index_finger_tip',
    'middle_finger_mcp',
    'middle_finger_pip',
    'middle_finger_dip',
    'middle_finger_tip',
    'ring_finger_mcp',
    'ring_finger_pip',
    'ring_finger_dip',
    'ring_finger_tip',
    'pinky_mcp',
    'pinky_pip',
    'pinky_dip',
    'pinky_tip',
]

# image
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# Load the input image from an image file.
mp_image = mp.Image.create_from_file('/Users/mahathimanda/Desktop/build18/E.png')

# Create a hand landmarker instance with the image mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)

with HandLandmarker.create_from_options(options) as landmarker:
  # The landmarker is initialized. 
    hand_landmarker_result = landmarker.detect(mp_image)
    for l, landmark in enumerate(hand_landmarker_result.hand_world_landmarks[0]):
        name = landmark_names[l]
        print(name, landmark.x, landmark.y, landmark.z, '\n')
    
"""
# Use OpenCV’s VideoCapture to load the input video.
video_path = 'path/to/your/video.mp4'
cap = cv2.VideoCapture(video_path)

# Load the frame rate of the video using OpenCV’s CV_CAP_PROP_FPS
# You’ll need it to calculate the timestamp for each frame.
fps = cap.get(cv2.CAP_PROP_FPS)

# Loop through each frame in the video using VideoCapture#read()
# Convert the frame received from OpenCV to a MediaPipe’s Image object.
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame received from OpenCV to a MediaPipe's Image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    hand_landmarker_result = landmarker.detect_for_video(mp_image, fps)

    # Display the frame if needed
    # cv2.imshow('Frame', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close any open windows
cap.release()
cv2.destroyAllWindows()

"""