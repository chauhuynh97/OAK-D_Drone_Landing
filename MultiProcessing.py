import cv2
import depthai as dai
import numpy as np
import blobconverter
import CoDrone
import queue
from threading import Thread
from multiprocessing import Process, Value, Queue, freeze_support
import time


def drone_thread_function(q, paired_flag, data_flag):

    # set up drone connection and take off before trying to land
    print('Creating Drone Object')
    drone = CoDrone.CoDrone()
    print("Getting Ready to Pair")
    drone.pair(drone.Nearest)
    print("Paired")
    paired_flag.value = 1
    data_flag.value = 1
    drone.takeoff()
    print("Taking Off")
    print(drone.get_height())

    while True:

        if data_flag.value == 0:
            start_coords = q.get(0)
            print('from drone',start_coords)
            straight_land(drone,start_coords)
            data_flag.value = 1

        # else:
        #     print('q is empty')



def camera_thread_function(q, paired_flag,data_flag):

    # camera setup
    syncNN = False

    # Create pipeline
    pipeline = dai.Pipeline()
    labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    # Define sources and outputs
    camRgb = pipeline.create(dai.node.ColorCamera)
    spatialDetectionNetwork = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)

    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutNN = pipeline.create(dai.node.XLinkOut)
    xoutBoundingBoxDepthMapping = pipeline.create(dai.node.XLinkOut)
    xoutDepth = pipeline.create(dai.node.XLinkOut)

    xoutRgb.setStreamName("rgb")
    xoutNN.setStreamName("detections")
    xoutBoundingBoxDepthMapping.setStreamName("boundingBoxDepthMapping")
    xoutDepth.setStreamName("depth")

    # Properties
    camRgb.setPreviewSize(300, 300)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    # Setting node configs
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

    spatialDetectionNetwork.setBlobPath(blobconverter.from_zoo(name='mobilenet-ssd', shaves=6))
    spatialDetectionNetwork.setConfidenceThreshold(0.5)
    spatialDetectionNetwork.input.setBlocking(False)
    spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
    spatialDetectionNetwork.setDepthLowerThreshold(100)
    spatialDetectionNetwork.setDepthUpperThreshold(5000)

    # Linking
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    camRgb.preview.link(spatialDetectionNetwork.input)

    if syncNN:
        spatialDetectionNetwork.passthrough.link(xoutRgb.input)
    else:
        camRgb.preview.link(xoutRgb.input)

    spatialDetectionNetwork.out.link(xoutNN.input)
    spatialDetectionNetwork.boundingBoxMapping.link(xoutBoundingBoxDepthMapping.input)

    stereo.depth.link(spatialDetectionNetwork.inputDepth)
    spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)

    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:

        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
        xoutBoundingBoxDepthMapping = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=4, blocking=False)
        depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

        counter = 0
        fps = 0

        while True:
            # xs,ys,zs = 0,0,0
            inPreview = previewQueue.get()
            inDet = detectionNNQueue.get()
            depth = depthQueue.get()

            frame = inPreview.getCvFrame()
            depthFrame = depth.getFrame()

            depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            depthFrameColor = cv2.equalizeHist(depthFrameColor)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

            detections = inDet.detections
            if len(detections) != 0:
                boundingBoxMapping = xoutBoundingBoxDepthMapping.get()
                roiDatas = boundingBoxMapping.getConfigData()

                for roiData in roiDatas:
                    roi = roiData.roi
                    roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
                    topLeft = roi.topLeft()
                    bottomRight = roi.bottomRight()
                    xmin = int(topLeft.x)
                    ymin = int(topLeft.y)
                    xmax = int(bottomRight.x)
                    ymax = int(bottomRight.y)

                    cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), 255, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

            # If the frame is available, draw bounding boxes on it and show the frame
            height = frame.shape[0]
            width = frame.shape[1]
            for detection in detections:
                # Denormalize bounding box
                x1 = int(detection.xmin * width)
                x2 = int(detection.xmax * width)
                y1 = int(detection.ymin * height)
                y2 = int(detection.ymax * height)
                try:
                    label = labelMap[detection.label]
                except:
                    label = detection.label
                cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, "{:.2f}".format(detection.confidence * 100), (x1 + 10, y1 + 35),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), cv2.FONT_HERSHEY_SIMPLEX)

                xs = int(detection.spatialCoordinates.x)
                ys = int(detection.spatialCoordinates.y)
                zs = int(detection.spatialCoordinates.z)


                if paired_flag.value == 1 and data_flag.value == 1:
                    q.put([xs,0,zs])
                    data_flag.value = 0

            cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4,
                        (255, 255, 255))

            cv2.imshow("preview", frame)

            if cv2.waitKey(1) == ord('q'):
                break


# simple straight landing function
def straight_land(drone, start_coords, land_coords = [0,0,0]):
    #startTime = time.time()

    height_threshold = 150 # height before drone.land()
    depth_threshold = 1500
    left_threshold = -300
    right_threshold = 300

    # x:left-right, y:height, z:depth in mm
    xs,ys,zs = start_coords
    xl,yl,zl = land_coords

    pitch_p = 30  # 20 is the scale factor for pitch, assume while loop repeat 3 times

    x = xs - xl
    z = zs - zl
    if z > depth_threshold:
        drone.set_roll(0)
        drone.set_pitch(pitch_p)
        drone.move(1)
    else:
        drone.set_pitch(0)
        drone.move(0)

    if x < left_threshold:
        drone.set_pitch(0)
        drone.set_roll(-25)
        drone.move(1)
    elif x > right_threshold:
        drone.set_pitch(0)
        drone.set_roll(25)
        drone.move(1)
    else:
        drone.set_roll(0)
        drone.move(0)

    if z <= depth_threshold and left_threshold < x < right_threshold:
        drone.go_to_height(height_threshold)

        # print('within landing region')

        print('coords before landing', start_coords)
        drone.set_pitch(15)
        drone.move(1)
        print('landing')
        drone.land()
        print("landing")
        drone.close()
        print('drone closed')

    # executionTime = (time.time() - startTime)
    # print('Execution time in seconds: ' + str(executionTime))


# pitch forward and continuously checking the z coords
def forward_land(drone, start_coords,land_coords=[0,0,0]):
    # depth distance in mm
    xs,ys,zs = start_coords
    xl,yl,zl = land_coords

    z = zs - zl
    pitch_power = 30
    if z > 1500:
        drone.set_pitch(pitch_power)
        drone.move(2)
    elif z > 1000:
        drone.set_pitch(pitch_power)
        drone.move(1)
    else:
        straight_land(drone,start_coords,land_coords)


if __name__ == '__main__':
    freeze_support()

    # exitFlag = 0
    paired_flag = Value('i',0)
    data_flag = Value('i',0)
    #camera_flag = Value('i',0)

    locationQ = Queue()

    camera = Process(target=camera_thread_function, args=(locationQ,paired_flag,data_flag))
    camera.start()

    drone1 = Process(target=drone_thread_function, args=(locationQ,paired_flag,data_flag))
    drone1.start()

    # while exitFlag == 0:
    #     pass

    camera.join()
    drone1.join()

    # print('length of q', len(locationQ))



