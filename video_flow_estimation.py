import cv2
import pafy
import numpy as np

from raft import Raft

FLOW_FRAME_OFFSET = 10 # Number of frame difference to estimate the optical flow

# Initialize model
model_path='models/raft_things_iter20_240x320.onnx'
flow_estimator = Raft(model_path)

# Initialize video
cap = cv2.VideoCapture("doc/eagle3.mp4")

# Skip first {start_time} seconds
start_time = 1
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time*30)

cv2.namedWindow("Estimated flow", cv2.WINDOW_NORMAL)
frame_list = []	
frame_num = 0
while cap.isOpened():

	try:
		# Read frame from the video
		ret, prev_frame = cap.read()
		frame_list.append(prev_frame)
		if not ret:	
			break
	except:
		continue

	# Skip the first frames to be able to 
	frame_num += 1
	if frame_num <= FLOW_FRAME_OFFSET:
		continue

	flow_map = flow_estimator(frame_list[0], frame_list[-1])
	flow_img = flow_estimator.draw_flow()

	alpha = 0.5
	combined_img = cv2.addWeighted(frame_list[0], alpha, flow_img, (1-alpha),0)
	# combined_img = np.hstack((frame_list[-1], flow_img))

	cv2.imshow("raw", flow_img)
	cv2.imshow("Estimated flow", combined_img)

	# Remove the oldest frame
	frame_list.pop(0)

	# Press key q to stop
	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()