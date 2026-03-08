import cv2
import numpy as np
import pandas as pd

input_video_path = "/content/merged.mp4"
cap = cv2.VideoCapture(input_video_path)

fps = 10
frame_count = 0
feature_data = []
initial_vertical_distance = None

video_fps = cap.get(cv2.CAP_PROP_FPS)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % int(video_fps / fps) == 0:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            cnt = max(contours, key=cv2.contourArea)

            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)

            M = cv2.moments(cnt)
            cx = M["m10"] / M["m00"] if M["m00"] else x + w / 2
            cy = M["m01"] / M["m00"] if M["m00"] else y + h / 2

            leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
            rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])

            hu_moments = cv2.HuMoments(M).flatten()
            hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area else 0

            extent = float(area) / (w * h) if w * h else 0

            aspect_ratio = float(w) / h if h else 0

            equiv_diameter = np.sqrt(4 * area / np.pi) if area else 0

            try:
                if len(cnt) >= 5:
                    ellipse = cv2.fitEllipse(cnt)
                    (center, axes, angle) = ellipse
                    major_axis, minor_axis = max(axes), min(axes)
                    eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
                else:
                    angle = 0
                    eccentricity = 0
            except:
                angle = 0
                eccentricity = 0

            gray_thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
            white_pixels = cv2.findNonZero(gray_thresh)

            if white_pixels is not None:
                coords = [tuple(pt[0]) for pt in white_pixels]

                min_x = min(coords, key=lambda p: p[0])[0]
                max_x = max(coords, key=lambda p: p[0])[0]

                base_ys = [p[1] for p in coords if p[0] == min_x]
                tip_ys = [p[1] for p in coords if p[0] == max_x]

                base_y = int((min(base_ys) + max(base_ys)) / 2)
                tip_y = int((min(tip_ys) + max(tip_ys)) / 2)

                current_vertical_distance = tip_y - base_y

                if initial_vertical_distance is None:
                    initial_vertical_distance = current_vertical_distance

                relative_deflection = current_vertical_distance - initial_vertical_distance
            else:
                relative_deflection = 0

            feature_data.append([
                cx, cy, x, y, w, h, area, perimeter,
                leftmost[0], leftmost[1], rightmost[0], rightmost[1],
                *hu_moments, solidity, extent, aspect_ratio,
                equiv_diameter, eccentricity, angle, relative_deflection
            ])

    frame_count += 1

cap.release()

columns = [
    "cx","cy","x","y","w","h","area","perimeter",
    "leftmost_x","leftmost_y","rightmost_x","rightmost_y",
    "hu1","hu2","hu3","hu4","hu5","hu6","hu7",
    "solidity","extent","aspect_ratio","equiv_diameter",
    "eccentricity","orientation_angle","tip_deflection"
]

features_df = pd.DataFrame(feature_data, columns=columns)
features_df.to_csv("feature.csv", index=False)

print("Features extracted and saved")
