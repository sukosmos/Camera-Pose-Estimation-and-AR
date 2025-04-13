import numpy as np
import cv2 as cv

def select_img_from_video(video_file, board_pattern, select_all=False, wait_msec=10, wnd_name='Camera Calibration'):
    video = cv.VideoCapture(video_file)
    assert video.isOpened(), f"Cannot open video: {video_file}"

    selected = []
    while True:
        valid, img = video.read()
        if not valid:
            break

        if select_all:
            selected.append(img)
        else:
            display = img.copy()
            cv.putText(display, f'#Selected: {len(selected)}', (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))
            cv.imshow(wnd_name, display)

            key = cv.waitKey(wait_msec)
            if key == ord(' '):  # Space: Show corners
                complete, pts = cv.findChessboardCorners(img, board_pattern)
                if complete:
                    cv.drawChessboardCorners(display, board_pattern, pts, complete)
                cv.imshow(wnd_name, display)
                key = cv.waitKey()
                if key == ord('\r'):  # Enter to select
                    selected.append(img)
            elif key == 27:  # ESC to finish
                break

    cv.destroyAllWindows()
    return selected

def calib_camera_from_chessboard(images, board_pattern, board_cellsize, K=None, dist_coeff=None, calib_flags=None):
    obj_points, img_points, gray_shape = [], [], None

    # Real-world coordinates for chessboard corners
    objp = np.zeros((board_pattern[0] * board_pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_pattern[0], 0:board_pattern[1]].T.reshape(-1, 2)
    objp *= board_cellsize

    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        complete, pts = cv.findChessboardCorners(gray, board_pattern)
        if complete:
            img_points.append(pts)
            obj_points.append(objp)
            gray_shape = gray.shape[::-1]
 
    assert len(img_points) > 0, "No chessboard was detected in the selected images!"
    assert gray_shape is not None

    return cv.calibrateCamera(obj_points, img_points, gray_shape, K, dist_coeff, flags=calib_flags)

if __name__ == '__main__':
    video_path = r'C:\Users\user\Documents\25-1\컴비\codes\Camera-Calibration\checkerboard.mp4'
    board_pattern = (10, 7)            # Number of internal corners (cols, rows)
    board_cellsize = 0.025             # Cell size in meters

    images = select_img_from_video(video_path, board_pattern, select_all = True)
    assert len(images) > 0, "No images selected!"

    rms, K, dist_coeff, rvecs, tvecs = calib_camera_from_chessboard(images, board_pattern, board_cellsize)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    print("\n## Camera Calibration Results")
    print(f"* The number of selected images = {len(images)}")
    print(f"* RMS error = {rms:.6f}")
    print(f"* fx = {fx:.6f}, fy = {fy:.6f}")
    print(f"* cx = {cx:.6f}, cy = {cy:.6f}")
    print(f"* Camera matrix (K):\n{K}")
    print(f"* Distortion coefficients (k1, k2, p1, p2, k3, ...) =\n{dist_coeff.flatten()}")

    # Optionally save results
    np.savez('calibration_result.npz', K=K, dist=dist_coeff)
    print("✅ Calibration result saved to 'calibration_result.npz'")

