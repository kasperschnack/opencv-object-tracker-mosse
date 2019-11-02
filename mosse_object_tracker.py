import time
import argparse

import cv2
import imutils
from imutils.video import VideoStream, FPS

tracker = cv2.TrackerMOSSE_create()

initBB = None

print("[INFO] starting video stream...")
video_stream = VideoStream(src=0).start()
time.sleep(1)

fps = None

while True:
    frame = video_stream.read()

    frame = imutils.resize(frame, width=500)
    height, width = frame.shape[:2]

    # Check to see if we are currently tracking an object
    if initBB is not None:
        success, box = tracker.update(frame)

        # check if tracking succeedes
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            fps.update()
            fps.stop()

            # initialize the set of information we'll be displaying on
            # the frame
            info = [
                ("Success", "Yes" if success else "No"),
                ("FPS", "{:.2f}".format(fps.fps())),
            ]

            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(
                    frame,
                    text,
                    (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 's' key is selected, we are going to "select" a bounding
    # box to track
    if key == ord("s"):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        initBB = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)

        # start OpenCV object tracker using the supplied bounding box
        # coordinates, then start the FPS throughput estimator as well
        tracker.init(frame, initBB)
        fps = FPS().start()
