# Video procesing
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
from gooey import Gooey, GooeyParser

# Data science
import seaborn as sns 
from matplotlib import pyplot as plt
import pandas as pd
sns.set_context("poster")
sns.axes_style()
sns.despine()
from scipy import stats

@Gooey(dump_build_config=True, program_name="MotionTracker")
def get_args():
	ap = GooeyParser(description="MotionLab but automatic.")
	ap.add_argument('Video File', widget='FileChooser')
	ap.add_argument('Width', default="-1")
	ap.add_argument(
        '-t', '--tracker', default="kcf", choices=['kcf', 'csrt', 'boosting', 'mil', 'tld', 'medianflow', 'mosse'], help='OpenCV object tracker type')
	return vars(ap.parse_args())

args=get_args()

OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.TrackerCSRT_create,
	"kcf": cv2.TrackerKCF_create,
	"boosting": cv2.TrackerBoosting_create,
	"mil": cv2.TrackerMIL_create,
	"tld": cv2.TrackerTLD_create,
	"medianflow": cv2.TrackerMedianFlow_create,
	"mosse": cv2.TrackerMOSSE_create
}
tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
initBB = None

if not args.get("Video File", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(1.0)
else:
	vs = cv2.VideoCapture(args["Video File"])

fps = None
(last_x, last_y) = (None, None)
data = {}

width = int(args.get("Width"))

frame_count = 1 

fout = open("data.csv", "w")
fout.write("frame,x_position,y_position,x_velocity,y_velocity\n")
while vs.isOpened():
	
	# grab the current frame, then handle if we are using a
	# VideoStream or VideoCapture object
	frame = vs.read()
	frame = frame[1] if args.get("Video File", False) else frame

	if frame is None:
            break;
        

	# resize the frame (so we can process it faster) and grab the
	# frame dimensions
	if width != -1:
		frame = imutils.resize(frame, width=width)
	(H, W) = frame.shape[:2]

	# check to see if we are currently tracking an object
	if initBB is not None:
		# grab the new bounding box coordinates of the object
		(success, box) = tracker.update(frame)
		if last_x is not None and last_y is not None:

			current_x = box[0]
			current_y = box[1]
			data[frame_count] = [current_x, current_y, current_x - last_x, current_y - last_y]
			fout.write(str(frame_count) + "".join(", " + str(item) for item in data[frame_count]) + "\n")
			last_x = current_x
			last_y = current_y
		else:
			last_x = box[0]
			last_y = box[1]

		# check to see if the tracking was a success
		if success:
			(x, y, w, h) = [int(v) for v in box]
			cv2.rectangle(frame, (x, y), (x + w, y + h),
				(0, 255, 0), 2)

		# update the FPS counter
		fps.update()
		fps.stop()

		# initialize the set of information we'll be displaying on
		# the frame
		info = [
			("Tracker", args["tracker"]),
			("FPS", "{:.1f}".format(fps.fps())),
		]

		# loop over the info tuples and draw them on our frame
		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(10) & 0xFF

	# if the 's' key is selected, we are going to "select" a bounding
	# box to track
	if key == ord("s"):
		# select the bounding box of the object we want to track (make
		# sure you press ENTER or SPACE after selecting the ROI)
		initBB = cv2.selectROI("Frame", frame, fromCenter=False,
			showCrosshair=True)

		# start OpenCV object tracker using the supplied bounding box
		# coordinates, then start the FPS throughput estimator as well
		tracker.init(frame, initBB)
		fps = FPS().start()

	# if the `q` key was pressed, break from the loop
	elif key == ord("q"):
		break

	frame_count += 1

# if we are using a webcam, release the pointer
if not args.get("Video File", False):
	vs.stop()
# otherwise, release the file pointer
else:
	vs.release()

# close all windows
cv2.destroyAllWindows()

# close csv 
fout.close()
for x in data:
	print(f"{x}: {data.get(x)}")

#### GRAPHING

df = pd.read_csv('data.csv')

for stat in ['x_position', 'y_position', 'x_velocity', 'y_velocity']:
	# get coeffs of linear fit
	slope, intercept, r_value, p_value, std_err = stats.linregress(df['frame'],df[stat])

	# use line_kws to set line label for legend
	ax = sns.regplot(x="frame", y=stat, data=df,
	line_kws={'label':"y={0:.1f}x+{1:.1f}".format(slope,intercept)})

	# plot legend
	ax.legend()

	sns.despine()
	plt.show()

	ax.get_figure().savefig(f"{stat}.png")
