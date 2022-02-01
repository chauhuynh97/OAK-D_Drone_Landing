# first, import all necessary modules
from pathlib import Path

import blobconverter
import cv2
import depthai
import numpy as np
import CoDrone

print('Creating Drone Object')
drone = CoDrone.CoDrone()
print ("Getting Ready to Pair")
drone.pair(drone.Nearest)

print("Paired")

drone.takeoff()
print("Taking Off")

print(drone.get_height())




# Pipeline tells DepthAI what operations to perform when running - you define all of the resources used and flows here
pipeline = depthai.Pipeline()

# First, we want the Color camera as the output
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(256, 256)  # 300x300 will be the preview frame size, available as 'preview' output of the node
cam_rgb.setInterleaved(False)

# Next, we want a neural network that will produce the detections
detection_nn = pipeline.createMobileNetDetectionNetwork()
# Blob is the Neural Network file, compiled for MyriadX. It contains both the definition and weights of the model
# We're using a blobconverter tool to retreive the MobileNetSSD blob automatically from OpenVINO Model Zoo
detection_nn.setBlobPath(blobconverter.from_zoo(name='face-detection-0200', shaves=6))
# Next, we filter out the detections that are below a confidence threshold. Confidence can be anywhere between <0..1>
detection_nn.setConfidenceThreshold(0.5)
# Next, we link the camera 'preview' output to the neural network detection input, so that it can produce detections
cam_rgb.preview.link(detection_nn.input)

# XLinkOut is a "way out" from the device. Any data you want to transfer to host need to be send via XLink
xout_rgb = pipeline.createXLinkOut()
# For the rgb camera output, we want the XLink stream to be named "rgb"
xout_rgb.setStreamName("rgb")
# Linking camera preview to XLink input, so that the frames will be sent to host
cam_rgb.preview.link(xout_rgb.input)

# The same XLinkOut mechanism will be used to receive nn results
xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)
# Straight landing function for drone
def straight_land(start_coords, land_coords = [0,0,0]):
    height_threshold = 150 # height before drone.land()
    # x:left-right, y:height, z:depth
    xs,ys,zs = start_coords
    xl,yl,zl = land_coords

    xs, ys, zs = xs*1000,ys*1000,zs*1000
    xl, yl, zl = xl*1000,yl*1000,zl*1000

    h = drone.get_height()  # relative to the ground
    print('Before pitch')
    print(h)

    pitch_p = zs/(20*3)  # 20 is the scale factor for pitch, assume while loop repeat 3 times
    # for now assume that camera is on the ground, drone is starting out higher than camera
    throttle_p = -50 #-ys/9  # 7 is the scale factor for throttle
    while h > height_threshold:
        drone.set_pitch(pitch_p)
        drone.move(1)

        h = drone.get_height()
        if h <= height_threshold:
            break

        throttle_p = throttle_p + 3
        if throttle_p > -15:
            drone.land()
            break
        else:
            drone.set_throttle(throttle_p)
            drone.move(1)
        # print(drone.get_height())

        h = drone.get_height()
        print('After throttle')
        print(h)

    drone.land()
    print("landing")

# Pipeline is now finished, and we need to find an available device to run our pipeline
# we are using context manager here that will dispose the device after we stop using it
with depthai.Device(pipeline) as device:
    # From this point, the Device will be in "running" mode and will start sending data via XLink

    # To consume the device results, we get two output queues from the device, with stream names we assigned earlier
    q_rgb = device.getOutputQueue("rgb")
    q_nn = device.getOutputQueue("nn")

    # Here, some of the default values are defined. Frame will be an image from "rgb" stream, detections will contain nn results
    frame = None
    detections = []

    # Since the detections returned by nn have values from <0..1> range, they need to be multiplied by frame width/height to
    # receive the actual position of the bounding box on the image
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


    # Main host-side application loop
    while True:

        # we try to fetch the data from nn/rgb queues. tryGet will return either the data packet or None if there isn't any
        in_rgb = q_rgb.tryGet()
        in_nn = q_nn.tryGet()

        if in_rgb is not None:
            # If the packet from RGB camera is present, we're retrieving the frame in OpenCV format using getCvFrame
            frame = in_rgb.getCvFrame()

        if in_nn is not None:
            # when data from nn is received, we take the detections array that contains mobilenet-ssd results
            detections = in_nn.detections

        if frame is not None:
            # for detection in detections:
            if len(detections) > 0:
                detection = detections[0]
                # for each bounding box, we first normalize it to match the frame size
                bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                # and then draw a rectangle on the frame to show the actual result
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                y = (detection.ymin+detection.ymax)/(2*100)
                # drone.hover(3)
                drone.land()
                # straight_land([0,0.8,1])
                print("Landing")
            # After all the drawing is finished, we show the frame on the screen
            cv2.imshow("preview", frame)



        # at any time, you can press "q" and exit the main loop, therefore exiting the program itself
        if cv2.waitKey(1) == ord('q'):
            break
        drone.close()
