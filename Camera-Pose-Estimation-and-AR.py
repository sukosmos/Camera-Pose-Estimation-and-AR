import numpy as np
import cv2 as cv
import math

# Load calibration result
calib_file = 'calibration_result.npz'
data = np.load(calib_file)
K = data['K']
dist_coeff = data['dist']

# Parameters
video_file = r'C:\Users\user\Documents\25-1\컴비\codes\Camera-Calibration\checkerboard.mp4'
board_pattern = (10, 7)
board_cellsize = 0.025
board_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
resize_scale = 0.5

# Function to generate 3D star points
def create_star_points(center, outer_r=0.5, inner_r=0.2, z=0):
    points = []
    for i in range(10):
        angle = math.pi / 5 * i  # 36 degrees per step
        r = outer_r if i % 2 == 0 else inner_r
        x = center[0] + r * math.cos(angle)
        y = center[1] + r * math.sin(angle)
        points.append([x, y, z])
    return np.array(points, dtype=np.float32)

# Create 3D star at center of chessboard
center_x = board_pattern[0] / 2
center_y = board_pattern[1] / 2
star_3d = board_cellsize * create_star_points(center=(center_x, center_y))

# Chessboard 3D points
obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])

# Open video
video = cv.VideoCapture(video_file)
assert video.isOpened(), "Cannot open video"

# Main loop
while True:
    valid, img = video.read()
    if not valid:
        break

    complete, img_points = cv.findChessboardCorners(img, board_pattern, None)
    if complete:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_points = cv.cornerSubPix(gray, img_points, (11, 11), (-1, -1), board_criteria)

        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

        # Project and draw the 3D star
        star_2d, _ = cv.projectPoints(star_3d, rvec, tvec, K, dist_coeff)
        star_2d = np.int32(star_2d).reshape(-1, 1, 2)
        cv.polylines(img, [star_2d], isClosed=True, color=(0, 255, 255), thickness=2)  # yellow

        # Display camera position
        R, _ = cv.Rodrigues(rvec)
        p = (-R.T @ tvec).flatten()
        info = f'XYZ: [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]'
        cv.putText(img, info, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    # Resize for display
    resized = cv.resize(img, None, fx=resize_scale, fy=resize_scale)
    cv.imshow("Pose Estimation with Star AR", resized)

    if cv.waitKey(30) == 27:  # ESC
        break

video.release()
cv.destroyAllWindows()

