import cv2
import numpy as np
from google.colab.patches import cv2_imshow

input_video_path = "/content/4.mp4"
output_video_path = "/content/T_4.mp4"

cap = cv2.VideoCapture(input_video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height), isColor=False)

lower_blue = np.array([100, 150, 0])
upper_blue = np.array([140, 255, 255])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    output_frame = np.zeros_like(blue_mask)
    output_frame[blue_mask == 255] = 255

    cv2_imshow(output_frame)

    out.write(output_frame)

cap.release()
out.release()

print(f"Output video saved to: {output_video_path}")
