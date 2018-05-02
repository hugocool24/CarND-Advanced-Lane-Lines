import numpy as np
import cv2
import os
from moviepy.editor import VideoFileClip

def calibrate_camera(img, directory = "camera_cal/", nx = 9, ny = 6):
	# Arrays to store information
	objectPoints = []  # 3D points in real-world space
	imagePoints = []  # 2D points in image-plane
	# Prepare object points
	# This part of the code is from the lectures
	objPoints = np.zeros((ny * nx, 3), np.float32)
	objPoints[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

	for image in os.listdir(directory):
		img = cv2.imread(directory + image)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# Find chessboard corners
		ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
		if ret:
			objectPoints.append(objPoints)
			imagePoints.append(corners)
	img_size = (img.shape[1], img.shape[0])
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints, img_size, None, None)
	return mtx, dist

def distortion_correction(mtx, dist, img):
	# Do camera calibration given object points and image points
	undist = cv2.undistort(img, mtx, dist, None, mtx)
	return undist

#This code of based of the code in Color and Gradient
def color_gradient(img):
	# This part of the code is a modified version of the code presented in the lecture 30 - Color and gradient
	# Note: img is the undistorted image
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	s_channel = hls[:,:,2]

	# Grayscale image
	# NOTE: we already saw that standard grayscaling lost color information for the lane lines
	# Explore gradients in other colors spaces / color channels to see what might work better
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

	# Sobel x
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
	abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
	scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

	# Threshold x gradient
	thresh_min = 20
	thresh_max = 100
	sxbinary = np.zeros_like(scaled_sobel)
	sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

	# Threshold color channel
	s_thresh_min = 170
	s_thresh_max = 255
	s_binary = np.zeros_like(s_channel)
	s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

	# Stack each channel to view their individual contributions in green and blue respectively
	# This returns a stack of the two binary images, whose components you can see as different colors
	color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

	# Combine the two binary thresholds
	combined_binary = np.zeros_like(sxbinary)
	combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
	return combined_binary

# This code is based of the code from lecture 17 - form a stop sign
def perspective_transform(img):

	offsetX1 = 200
	offsetX2 = 590
	src = [[offsetX1, 690],
		   [offsetX2, 450],
		   [1280 - offsetX2, 450],
		   [1280 - offsetX1, 690]]

	src = np.float32(src)

	offsetX = 200
	dst = [[offsetX, 720],
		   [offsetX, 0],
		   [1280 - offsetX, 0],
		   [1280 - offsetX, 720]]
	dst = np.float32(dst)
	img_size = (img.shape[1], img.shape[0])
	M = cv2.getPerspectiveTransform(src, dst)
	Minv = cv2.getPerspectiveTransform(dst, src)
	warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
	return warped, Minv

#This code is based of the lecture about Finding the lines
def reset_sliding_window(binary_warped):
	# Assuming you have created a warped binary image called "binary_warped"
	# Take a histogram of the bottom half of the image
	histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
	# Create an output image to draw on and  visualize the result
	out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0] // 2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# Choose the number of sliding windows
	nwindows = 9
	# Set height of windows
	window_height = np.int(binary_warped.shape[0] // nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	# Set the width of the windows +/- margin
	margin = 100
	# Set minimum number of pixels found to recenter window
	minpix = 50
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
		win_y_low = binary_warped.shape[0] - (window + 1) * window_height
		win_y_high = binary_warped.shape[0] - window * window_height
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin
		# Draw the windows on the visualization image
		cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
					  (0, 255, 0), 2)
		cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
					  (0, 255, 0), 2)
		# Identify the nonzero pixels in x and y within the window
		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
						  (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
						   (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
		# Append these indices to the lists
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)
		# If you found > minpix pixels, recenter next window on their mean position
		if len(good_left_inds) > minpix:
			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > minpix:
			rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)

	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
	left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
	right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

	return left_fit, right_fit, leftx, lefty, rightx, righty, ploty, left_fitx, right_fitx

#This code is based of the lecture about Finding the lines
def sliding_window(left_fit, right_fit):
	# Assume you now have a new warped binary image
	# from the next frame of video (also called "binary_warped")
	# It's now much easier to find line pixels!
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	margin = 100
	left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
								   left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
																		 left_fit[1] * nonzeroy + left_fit[
																			 2] + margin)))

	right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
									right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
																		   right_fit[1] * nonzeroy + right_fit[
																			   2] + margin)))

	# Again, extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]
	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
	left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
	right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
	return leftx, lefty, rightx, righty, ploty, left_fitx, right_fitx

#This part is influenced by the code in Measuring Curvature
def measure_curvature(leftx, lefty, rightx, righty, ploty):
	ym_per_pix = 30 / 710  # meters per pixel in y dimension
	xm_per_pix = 3.7 / 800 # meters per pixel in x dimension
	y_eval = np.max(ploty)
	left_fit_cr = np.polyfit(lefty * ym_per_pix, xm_per_pix * leftx, 2)
	right_fit_cr = np.polyfit(righty * ym_per_pix, xm_per_pix * rightx, 2)

	left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
		2 * left_fit_cr[0])
	right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
		2 * right_fit_cr[0])
	return left_curverad, right_curverad, left_fit_cr, right_fit_cr

#This part is based of the code from "Tips and Tricks for the Project"
def plot_lines(image, undist, left_fitx, right_fitx, ploty, warped, Minv, left_fit_cr, right_fit_cr, left_curve, right_curve):
	# Create an image to draw the lines on
	warp_zero = np.zeros_like(warped).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
	# Combine the result with the original image
	result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
	# Get distance to lanes from center of the car
	car_offset = offset_from_center(image, left_fit_cr, right_fit_cr)
	cv2.putText(result, car_offset,
				(10, 30),
				cv2.FONT_HERSHEY_SIMPLEX,
				1,
				(255,255,255),
				2)
	cv2.putText(result, "Curvature of left curve in m: "+str(round(left_curve,2)),
				(10, 70),
				cv2.FONT_HERSHEY_SIMPLEX,
				1,
				(255,255,255),
				2)
	cv2.putText(result, "Curvature of right curve in m: "+str(round(right_curve,2)),
				(10, 110),
				cv2.FONT_HERSHEY_SIMPLEX,
				1,
				(255,255,255),
				2)
	return result

def offset_from_center(image, left_fit_cr, right_fit_cr):
	ym_per_pix = 30 / 710  # meters per pixel in y dimension
	xm_per_pix = 3.7 / 800 # meters per pixel in x dimension
	#The middle point where the car is
	midpoint = (image.shape[1] / 2) * xm_per_pix
	# Calculate where the middle of the lane is
	constant = image.shape[0] * ym_per_pix
	left_line_start = left_fit_cr[0]*constant**2 + left_fit_cr[1]*constant + left_fit_cr[2]
	right_line_start = right_fit_cr[0]*constant**2 + right_fit_cr[1]*constant + right_fit_cr[2]
	#The middle of the lanes
	midLane = (left_line_start + right_line_start) / 2
	car_offset = midpoint - midLane
	if car_offset > 0:
		return "Car offset from midpoint: "+str(round(car_offset,2)) + "m to the right"
	else:
		return "Car offset from midpoint: " + str(round(-car_offset, 2)) + "m to the left"

def sanity_check(left_fit_cr, right_fit_cr):
	#Returns True of the check is NOK return False of check is OK
	#Check they are seperated by roughly the same distance horizontaly within 5%
	#Calculate from the start of the line
	constant1 = img.shape[0]
	#Calculate from the end of the line
	constant2 = 450
	left_line_start = left_fit_cr[0] * constant ** 2 + left_fit_cr[1] * constant + left_fit_cr[2]
	right_line_start = right_fit_cr[0] * constant ** 2 + right_fit_cr[1] * constant + right_fit_cr[2]
	left_line_end = left_fit_cr[0] * constant ** 2 + left_fit_cr[1] * constant + left_fit_cr[2]
	right_line_end = right_fit_cr[0] * constant ** 2 + right_fit_cr[1] * constant + right_fit_cr[2]
	distance1 = right_line_start - left_line_start
	distance2 = right_line_end - left_line_end
	if distance1 > 1.05*distance2 or distance1 < 0.95*distance2:
		return True
	else:
		return False



def main(img):
	#Correct the image distortion
	global mtx
	global dist
	undist = distortion_correction(mtx, dist, img)
	#Threshold the image
	combined_binary = color_gradient(undist)

	#Perspective transformation
	binary_warped, Minv = perspective_transform(combined_binary)

	#Run a sliding window. If the sanity-check fails then run reset sliding window else run sliding window
	#left_fit, right_fit = reset_sliding_window(binary_warped)
	try:
		leftx, lefty, rightx, righty, ploty, left_fitx, right_fitx = sliding_window(left_fit, right_fit)
		left_curve, right_curve, left_fit_cr, right_fit_cr = measure_curvature(leftx, lefty, rightx, righty, ploty)
		if sanity_check(left_fit_cr, right_fit_cr):
			left_fit, right_fit, leftx, lefty, rightx, righty, ploty, left_fitx, right_fitx = reset_sliding_window(binary_warped)
	except:
		left_fit, right_fit, leftx, lefty, rightx, righty, ploty, left_fitx, right_fitx = reset_sliding_window(binary_warped)
		left_curve, right_curve, left_fit_cr, right_fit_cr = measure_curvature(leftx, lefty, rightx, righty, ploty)

	#Plot the image
	result = plot_lines(img, undist, left_fitx, right_fitx, ploty, binary_warped, Minv, left_fit_cr, right_fit_cr, left_curve, right_curve)
	return result

#First calibrate the camera!
img = cv2.imread('test_images/test1.jpg')
mtx, dist = calibrate_camera(img)

clips = ('project_video.mp4','challenge_video.mp4','harder_challenge_video.mp4')
for clip in clips:
	clip1 = VideoFileClip(clip)
	write_output = 'output_video/' + clip
	write_clip = clip1.fl_image(main) #NOTE: this function expects color images!!
	write_clip.write_videofile(write_output, audio=False)

