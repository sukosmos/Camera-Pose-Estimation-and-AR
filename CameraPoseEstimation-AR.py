import numpy as np
import cv2 as cv
import math

# Load calibration data
data = np.load('calibration_result.npz')
K = data['K']
dist_coeff = data['dist']

# Video and chessboard settings
video_file = r'C:\Users\user\Documents\25-1\컴비\codes\Camera-Calibration\checkerboard.mp4'
board_pattern = (10, 7)
board_cellsize = 0.025
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Function to create 2D 10-pointed star on z=0 plane
def create_star_points(center, outer_r=0.5, inner_r=0.2, z=0.0):
    points = []
    for i in range(10):
        angle = math.pi / 5 * i
        r = outer_r if i % 2 == 0 else inner_r
        x = center[0] + r * math.cos(angle)
        y = center[1] + r * math.sin(angle)
        points.append([x, y, z])
    return np.array(points, dtype=np.float32)

# Center of chessboard area (not corner)
center_x = (board_pattern[0] - 1) / 2
center_y = (board_pattern[1] - 1) / 2
star_3d = board_cellsize * create_star_points(center=(center_x, center_y), z=0.0)

# 3D points of chessboard corners
obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])

# Open video
video = cv.VideoCapture(video_file)
assert video.isOpened(), "Cannot open video"

while True:
    valid, img = video.read()
    if not valid:
        break

    found, corners = cv.findChessboardCorners(img, board_pattern, None)
    if found:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Estimate pose
        ret, rvec, tvec = cv.solvePnP(obj_points, corners, K, dist_coeff)

        # Project star points
        star_2d, _ = cv.projectPoints(star_3d, rvec, tvec, K, dist_coeff)
        star_2d = np.int32(star_2d).reshape(-1, 1, 2)

        # Draw filled yellow star (★)
        cv.fillPoly(img, [star_2d], color=(0, 255, 255))  # Yellow fill
        cv.polylines(img, [star_2d], isClosed=True, color=(0, 200, 200), thickness=1)  # Outline

        # Display camera position
        R, _ = cv.Rodrigues(rvec)
        p = (-R.T @ tvec).flatten()
        info = f'XYZ: [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]'
        cv.putText(img, info, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    # Display original size
    cv.imshow("2D Star AR", img)

    if cv.waitKey(30) == 27:
        break

video.release()
cv.destroyAllWindows()


