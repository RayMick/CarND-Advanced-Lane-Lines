## Advanced Lane Finding

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistorted.png "Undistorted"
[image2]: ./output_images/undistorted_test2.png "Road Transformed"
[image3]: ./output_images/combined_binary_image.png "Binary Example"
[image4]: ./output_images/unwarped.png "Warp Example"
[image5]: ./output_images/binary_unwarped.png "Fit Visual"
[image6]: ./output_images/polyfit.png "Output"
[image7]: ./output_images/formula_curvature.PNG "Output"
[image8]: ./output_images/draw_results_back.png "Output"
[image9]: ./output_images/initial_saturation_red_combine.PNG "Output"
[image10]: ./output_images/masked_saturation_red_combine.PNG "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how each one is addressed.    

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the 3rd code cell of the IPython notebook located in current directory named as "Advanced_Lane_Finding.ipynb".

I start by preparing "object points", which will be the inner corners of the chessboard in the world (`objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)`). Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the pixel position of each of the corners in the image plane with each successful chessboard detection (`ret, corners = cv2.findChessboardCorners(gray, (9,6),None)`).  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images (./test_images/test2.jpg). After the camera is calibrated, the calibration results are saved
```Python
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open( "camera_cal/calibration_pickle.p", "wb" ) )```

And then the "test2,jpg" is read in and `cv2.undistort` is called to do the undistortion. The result is show in the image below.

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

In order to achieve good detection results of both the yellow lane and dashed white line, lots of different transformations are experimented. The following unit functions are defined in the code, each of them implements an independent transformation.
* Absolute Sobel Gradient  
`def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):`
* Sobel Magnitude  
`def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):`
* Thresholded Gradient Direction  
`def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):`
* HLS Color Space Selection  
`def hls_select(img, thresh=(0, 255)):`
* RGB Color Space Selection  
`def red_select(img, thresh=(0, 255)):`

Various combinations of the above functions are also tried (such as abs_sobel_thresh + mag_binary + dir_binary   or  HSL+abs_sobel_thresh or HSL + RGB ).
It turns out that combining the saturation and red channel filters provides the most satisfying results. The code snippet is shown below:
```Python
red_binary = red_select(image, thresh=(235, 255))
hls_binary = hls_select(image, thresh=(115, 255))
color_binary = np.dstack((red_binary, np.zeros_like(red_binary), hls_binary)) * 255

# Combine the two binary thresholds
combined_binary = np.zeros_like(red_binary)
combined_binary[(hls_binary == 1) | (red_binary == 1)] = 1
```
![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `unwarp()`, which appears in the 8th code cell of the IPython notebook.  The `unwarp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in as follows (manually measured in the image):

| Source        | Destination   |
|:-------------:|:-------------:|
| 581, 460      | 300, 0        |
| 705, 460      | 980, 0      |
| 279, 675     | 300, 720      |
| 1042, 675      | 980, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

Also the after I obtained the combined_binary image, a unwarp operaiton is carried out, the result of which is displayed below.
```Python
combined_binary_undist = cv2.undistort(combined_binary, mtx, dist, None, mtx)
combined_binary_undist_warped, M, M_inv = unwarp(combined_binary_undist, src, dst)
```

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

After applying calibration, thresholding, and a perspective transform to a road image, I obtained a binary image where the lane lines stand out clearly.
I then take a histogram along all the columns in the lower half of the image. With this histogram I am adding up the pixel values along each column in the image. In my thresholded binary image, pixels are either 0 or 1, so the two most prominent peaks in this histogram will be good indicators of the x-position of the base of the lane lines. I can use that as a starting point for where to search for the lines. From that point, I can use a sliding window, placed around the line centers, to find and follow the lines up to the top of the frame.
This part of code is implemented in the function called
`def histogram_polyfit(image)` in the jupyter notebook

![alt text][image6]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

After locating the lane line pixels and calcuating the x and y pixel positions to fit a second order polynomial curve, The radius of curvature at any point x of the function x=f(y) is given as follows:
![alt text][image7]

The code is implemented in the function `def curvature(binary_img, left_fit, right_fit, left_lane_inds, right_lane_inds)` , which takes in a binary images and left, right fit results and lane indices.

The curvature of this particular image is:

| Left curvature   | Right curvature   |
|:-------------:|:-------------:|
| 697.772189503 m      | 1756.30813074 m        |


Besides, a distance to the lane center function is also implemented and named as: `def center_dist(binary_img, left_fit, right_fit)`. This particular image, the distance to center is -0.342860918282 m.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented a function named `def draw_results(orig_img, binary_img, left_fit, right_fit, M_inv, curve, center_distance)` in my code in `Advanced_Lane_Finding.ipynb`.  This function will take in as arguments, original image, binary image after all pre-processings, lef/right fits, inversion matrix, curvature and distance to lane center. Here is an example of my result on a test image:

![alt text][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The most difficult part in this project is to generate a reasonably good binary image using the combination of the pre-precosssing filters. As mentioned earlier in this report, quite a few different pre-processing combinations have been experimented (abs_sobel_thresh + mag_binary + dir_binary   or  HSL+abs_sobel_thresh or HSL + RGB).
The saturation channel and red channel combination seems most promising. However, this combination is not very efficient in isolating the pixels for the white lane. It turns out that some background white pixels in the adjacent lane would "disrupt" the white lane detection if just using the "saturation + red" combination. After close examination of the individual output of the saturation and red channel output, I realize that the "noise" pixels near the white lane is generated by the saturation filter. Therefore, instead of doing a straightforward stacking of the two channel images (`combined_binary[(hls_binary == 1) | (red_binary == 1)] = 1`), I used a half-image-size mask to mask out the right half of the output from the saturation filter as shown below.
```Python
l_r_midpoint = 660
w = hls_binary.shape[1]
hls_binary[:, l_r_midpoint:w] = 0
```
The combined binary image before and after using this half-image mask are shown below in top and bottom row respectively.

![alt text][image9]
![alt text][image10]

As can be seen from this comparison, the white noisy pixel "blob" between right lane and the car is gone after this treatment.

In addition to the above procedure, a pool of fitted results is also maintained to keep track of the fitted lanes from the last few frames, which is shown in the function `def update_fit(self, fit, inds)` inside `class Line()`.

Beside these techniques, other possible methods can also be tried to improve the robustness of the results, such as enforce the equal distance between two lanes, try different color space to do more combinations of the filters to generate binary images. However, due to the time constraint, these ideas may be experimented during future revisits.
