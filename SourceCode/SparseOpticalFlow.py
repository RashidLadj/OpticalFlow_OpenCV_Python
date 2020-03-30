import cv2 as cv
import numpy as np

# Les parametres de l'estimation du flot optique de Lucas-Kanade
lk_params = dict(winSize = (15,15), maxLevel = 2, criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

cap = cv.VideoCapture("Video01.mp4")
# Couleur our afficher le flot optique
color = (255, 0, 0)
ret, first_frame = cap.read()
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
# Finds the strongest corners in the first frame by Shi-Tomasi method - we will track the optical flow for these corners
# https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html#goodfeaturestotrack
# Les parametres de detection de points caractéristiques (coins) de Shi-Tomasi qui est une amélioration de Harris mais on peut tester avec les deux

ifHarris = 0 # 1=Shitomasi 0 Harris
feature_params = dict(maxCorners = 300, qualityLevel = 0.02, minDistance = 50, blockSize = 7, useHarrisDetector = ifHarris, k = 0.04)
prev = cv.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)

mask = np.zeros_like(first_frame)

while(cap.isOpened()):
    # cv.waitKey(0)
    ret, frame = cap.read()
    if(not ret):
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Calcule du flot optique par la methode de Lucas-Kanade
    # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowpyrlk
    next, status, error = cv.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)
    # Selects good feature points for previous position
    good_old = prev[status == 1]
    # Selects good feature points for next position
    good_new = next[status == 1]
    # Draws the optical flow tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        # Returns a contiguous flattened array as (x, y) coordinates for new point
        a, b = new.ravel()
        # Returns a contiguous flattened array as (x, y) coordinates for old point
        c, d = old.ravel()
        # Draws line between new and old position with green color and 2 thickness
        mask = cv.line(mask, (a, b), (c, d), color, 2)
        # Draws filled circle (thickness of -1) at new position with green color and radius of 3
        frame = cv.circle(frame, (a, b), 3, color, -1)
    # Overlays the optical flow tracks on the original frame
    output = cv.add(frame, mask)
    # Updates previous frame
    prev_gray = gray.copy()
    # Updates previous good feature points
    prev = good_new.reshape(-1, 1, 2)
    # Opens a new window and displays the output frame
    cv.imshow("sparse optical flow", output)
    # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
    if cv.waitKey(10) & 0xFF == ord('q'):
        break
# The following frees up resources and closes all windows
cap.release()
cv.destroyAllWindows()