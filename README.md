# Camera-Pose-Estimation-and-AR
using openCV, put a star on the checkerboard

The star appears fixed on the center of the board in the camera’s coordinate system.

<br>

###  Features

- **Pose Estimation from a Chessboard**

```python
ret, rvec, tvec = cv.solvePnP(obj_points, corners, K, dist_coeff)

```

Estimates the camera’s rotation and translation vectors using the chessboard corners.

<br>

- ⭐ **2D Filled Star Rendering (★)**

```python
cv.fillPoly(img, [star_2d], color=(0, 255, 255))  # Yellow filled star

```

Projects a 10-pointed star onto the image plane and fills it to create a flat yellow star aligned with the chessboard.

<br>

### Input

- `checkerboard.mp4`: A video containing a 10×7 internal corner chessboard pattern.
- `calibration_result.npz`: File generated from camera calibration (`camera_calibration.py`)

<br><br>

## Result
<p align="center">
  <img src = "https://github.com/user-attachments/assets/8196c71d-4c61-4790-9441-70827ee8e5e5" width="40%" height="40%">  
</p>

- A solid yellow star (★) appears in the center of the chessboard in real-time.
- The star stays fixed in the world coordinate as the camera moves.
- Camera position is printed on the video frame (X, Y, Z).

<br><br>

---
*For Computer Vision*
