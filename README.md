# OpticalFlow_OpenCV_Python Sparse and Dense Optical-Flow
Initial video was taken from [here](https://www.pexels.com/video/cars-on-the-road-854745/)  
![alt text](https://media.giphy.com/media/LrX95dL1DsDPxYKi6V/giphy.gif)

# Sparse-Optical-Flow
The goal is to calculate sparse optical flow for a video with moving cars using Lucas-Kanade algorithm. The actual calculation will be performed by **cv2.calcOpticalFlowPyrLK**
You can find more info about this function in opencv [documentation](https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323)
For the choice of the key pixels to follow, the Harris or Shi-Tomasi method can be used.
If you want to read more about algorithm we will use, please, refer to [Pyramidal Implementation of the Ane Lucas Kanade Feature TrackerDescription of the algorithm](https://pdfs.semanticscholar.org/aa97/2b40c0f8e20b07e02d1fd320bc7ebadfdfc7.pdf) paper.  
![alt text](https://media.giphy.com/media/RJ1HpGFPAo0gKPAGBs/giphy.gif)

# Dense-Optical-Flow
The goal is to calculate dense optical flow for a video with moving cars using Gunnar Farneback algorithm.
The actual calculation will be performed by **cv2.calcOpticalFlowFarneback()**  
You can find more info about this function in opencv [documentation](https://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback)

If you want to read more about algorithm we will use, please, refer to [Two-Frame Motion Estimation Based on Polynomial Expansion](http://www.diva-portal.org/smash/get/diva2:273847/FULLTEXT01.pdf) paper.  
We also need to calculate magnitude and angle of 2D vectors, using **cv2.cartToPolar()**.    
Then we apply these values to the mask, so we could have different colors for different colors for different movement direction.  
Finally, we will combine mask and initial frame to get dense optical flow image.  
![alt text](https://media.giphy.com/media/kyiUNnxHG2TKMOjZd1/giphy.gif)
