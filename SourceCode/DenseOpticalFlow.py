import cv2
import numpy as np
from matplotlib import pyplot as plt

# Recuperer la video avec VideoCapture de OpenCV est la mettre dans la variable videoCapture
videoCapture = cv2.VideoCapture("Video01.mp4")
# Lire la premiere frame
ret, first_frame = videoCapture.read()
# Definir un certaine taille de l'image en gardant les perspective, utilisanation d'un scale
resize_dim = 600
max_dim = max(first_frame.shape)
scale = resize_dim/max_dim
first_frame = cv2.resize(first_frame, None, fx=scale, fy=scale)
# Convertir l'image au niveau du gris
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Create mask
mask = np.zeros_like(first_frame)
# Sets image saturation to maximum
mask[..., 1] = 255

while(videoCapture.isOpened()):
    #  Lire la prochaine frame, la resize et la convertir au nievau de gris
    ret, frame = videoCapture.read()
    if(not ret):
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=scale, fy=scale)

    # Calcul du flot optique dense en se basant sur l'algo de Farneback method
    # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, pyr_scale = 0.5, levels = 5, winsize = 11, iterations = 5, poly_n = 5, poly_sigma = 1.1, flags = 1)
    # Compute the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # Set image hue according to the optical flow direction
    mask[..., 0] = angle * 180 / np.pi / 2
    # Set image value according to the optical flow magnitude (normalized)
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    # Convertir de l'espace HSV to RGB
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    
    # resize de l'image en cours
    frame = cv2.resize(frame, None, fx=scale, fy=scale)
    
    # afficher l'image output
    dense_flow = cv2.addWeighted(frame, 1,rgb, 2, 0)
    #cv2.imshow("Dense optical flow", rgb)



    numpy_horizontal = np.hstack((frame, rgb, dense_flow))
    numpy_horizontal_concat = np.concatenate((frame, rgb, dense_flow), axis=1)
    cv2.imshow("Dense optical flow", numpy_horizontal_concat)

    # Mise a jour dans la frame src a trater pendant la prochaine itteration
    prev_gray = gray
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
videoCapture.release()

cv2.destroyAllWindows()





