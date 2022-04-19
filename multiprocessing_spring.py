import cv2
import depthai as dai
import numpy as np
import blobconverter
import CoDrone
import queue
from threading import Thread
from multiprocessing import Process, Value, Queue, freeze_support
import time

height_threshold = 180  # height before drone.land()
depth_upper_threshold = 1000
depth_lower_threshold = 300
left_threshold = -300
right_threshold = 300

drone_movement_start = 0
drone_movement_end = 0
data_request_start = 0
data_request_end = 0

def drone_thread_function(q,data_flag):

    # set up drone connection and take off before trying to land
    print('Creating Drone Object')
    drone = CoDrone.CoDrone()
    print("Getting Ready to Pair")
    drone.pair(drone.Nearest)
    print("Paired")
    data_flag.value = 1
    drone.takeoff()
    drone.set_throttle(50)
    drone_start_timer = time.time()
    print("Taking Off")
    print(drone.get_height())

    data_request_timer_counter = 0

    while True:

        if data_flag.value == 0 and not q.empty():
            start_coords = q.get(0)
            data_request_end = time.time()
            if data_request_timer_counter != 0:
                print('data request time', data_request_end - data_request_start)
            data_request_timer_counter += 1
            print('from drone',start_coords)
            # x:left-right, y:height, z:depth in mm
            xs,ys,zs = start_coords
            xl,yl,zl = [0,0,0]
            x = xs - xl
            z = zs - zl
            drone_movement_start = time.time()
            drone_land(drone,x,z,drone_start_timer)
            drone_movement_end = time.time()
            print('drone movement time', drone_movement_end - drone_movement_start)
            data_flag.value = 1
            data_request_start = time.time()

        # else:
        #     print('q is empty')

    data_flag.value = 0

def drone_thread_function2(q,detected_flag,data_flag):
    # set up drone connection and take off before trying to land
    print('Creating Drone Object')
    drone = CoDrone.CoDrone()
    print("Getting Ready to Pair")
    # drone.pair(drone.Nearest)
    # drone.pair('1484','COM6')
    drone.pair()
    print("Paired")
    # data_flag.value = 1
    detected_flag.value = 0
    drone.takeoff()
    # drone.set_throttle(50)
    # drone_start_timer = time.time()
    print("Taking Off")
    # print(drone.get_height())
    # drone.go_to_height(400)

    drone.set_pitch(20)

    while True:
        drone.move()
        data_flag.value = 1
        if detected_flag.value == 1:
            drone_land2(drone,q,detected_flag,data_flag)
            break
        # if detected_flag.value == 1 and not q.empty():
        #     data_flag.value = 1
        #     start_coords2 = q.get(0)
        #     xl,yl,zl = [0,0,0]
        #     xs,ys,zs = start_coords2
        #     x = xs - xl
        #     z = zs - zl
        #     drone_land2(drone,x,z)
        #     break

def drone_land2(drone,q,detected_flag,data_flag):
    print('in drone land')
    while True:
        if detected_flag.value == 1 and not q.empty():
            data_flag.value = 1
            start_coords2 = q.get(0)
            xl,yl,zl = [0,0,0]
            xs,ys,zs = start_coords2
            x = xs - xl
            z = zs - zl

            print(x,z)

            if x == 0 and z == 0:
                drone.set_pitch(20)
                drone.move(4)

            if depth_lower_threshold <= z <= depth_upper_threshold and left_threshold <= x <= right_threshold:
                drone.go_to_height(200)
                drone.land()
                drone.close()
                print('drone closed')
                drone.disconnect()
                print('drone disconnected')
                break
            else:
                drone_move(drone,x,z)

# def camera_thread_function(q,data_flag):
def camera_thread_function(q,detected_flag,data_flag):

    # camera setup
    labelMap = [
        "person",         "bicycle",    "car",           "motorbike",     "aeroplane",   "bus",           "train",
        "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",   "parking meter", "bench",
        "bird",           "cat",        "dog",           "horse",         "sheep",       "cow",           "elephant",
        "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",    "handbag",       "tie",
        "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball", "kite",          "baseball bat",
        "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",      "wine glass",    "cup",
        "fork",           "knife",      "spoon",         "bowl",          "banana",      "apple",         "sandwich",
        "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",       "donut",         "cake",
        "chair",          "sofa",       "pottedplant",   "bed",           "diningtable", "toilet",        "tvmonitor",
        "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",  "microwave",     "oven",
        "toaster",        "sink",       "refrigerator",  "book",          "clock",       "vase",          "scissors",
        "teddy bear",     "hair drier", "toothbrush"
    ]

    syncNN = True

    # Create pipeline
    pipeline = dai.Pipeline()

    # Define sources and outputs
    camRgb = pipeline.create(dai.node.ColorCamera)
    spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
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
    camRgb.setPreviewSize(416, 416)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    # setting node configs
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    #stereo.setExtendedDisparity(False)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)

    spatialDetectionNetwork.setBlobPath(blobconverter.from_zoo(name='yolo-v4-tiny-tf', shaves=6))
    spatialDetectionNetwork.setConfidenceThreshold(0.5)
    spatialDetectionNetwork.input.setBlocking(False)
    spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
    spatialDetectionNetwork.setDepthLowerThreshold(100)
    spatialDetectionNetwork.setDepthUpperThreshold(5000)

    # Yolo specific parameters
    spatialDetectionNetwork.setNumClasses(80)
    spatialDetectionNetwork.setCoordinateSize(4)
    spatialDetectionNetwork.setAnchors(np.array([10,14, 23,27, 37,58, 81,82, 135,169, 344,319]))
    spatialDetectionNetwork.setAnchorMasks({ "side26": np.array([1,2,3]), "side13": np.array([3,4,5]) })
    spatialDetectionNetwork.setIouThreshold(0.5)

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

        startTime = time.monotonic()
        counter = 0
        fps = 0
        color = (255, 255, 255)

        while True:
            person_detection_timer_start = time.time()

            # xs,ys,zs = 0,0,0
            inPreview = previewQueue.get()
            inDet = detectionNNQueue.get()
            depth = depthQueue.get()

            frame = inPreview.getCvFrame()
            depthFrame = depth.getFrame()

            depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            depthFrameColor = cv2.equalizeHist(depthFrameColor)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

            counter+=1
            current_time = time.monotonic()
            if (current_time - startTime) > 1 :
                fps = counter / (current_time - startTime)
                counter = 0
                startTime = current_time

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

                # if str(label) == 'person':
                #     person_detection_timer_end = time.time()
                    # print('Time taken to detect a person:', person_detection_timer_end - person_detection_timer_start)

                xs = int(detection.spatialCoordinates.x)
                ys = int(detection.spatialCoordinates.y)
                zs = int(detection.spatialCoordinates.z)

                # str(label) == "person" and
                # if data_flag.value == 1 and str(label) == "person":
                #     q.put([xs,0,zs])
                #     data_flag.value = 0
                #     # data_request_end = time.time()
                #     # print('data request time', data_request_end - data_request_start)

                if str(label) == "person":
                    detected_flag.value = 1
                    if data_flag.value == 1:
                        q.put([xs,0,zs])
                        data_flag.value = 0
                    # if counter % 50 == 0:
                    #     print("camera detected person")

            cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4,
                        (255, 255, 255))

            cv2.imshow("preview", frame)

            if cv2.waitKey(1) == ord('q'):
                break

def drone_move(drone, x,z):
    #startTime = time.time()

    pitch_power = 20                                                                                                        # 20 is the scale factor for pitch, assume while loop repeat 3 times
    roll_power = 10                                                                                                                 # moves left 20

    if x < left_threshold:
        drone.set_pitch(0)
        drone.set_throttle(0)
        print('move right')
        drone.set_roll(-roll_power)
        drone.move(1)
    elif x > right_threshold:
        drone.set_pitch(0)
        drone.set_throttle(0)
        print('move left')
        drone.set_roll(roll_power)
        drone.move(1)
    else:
        drone.set_roll(0)
        drone.set_throttle(0)
        drone.move(0)
        if z > depth_upper_threshold:
            drone.set_roll(0)
            drone.set_throttle(0)
            print('move forward')
            drone.set_pitch(pitch_power)
            drone.move(1)
        # elif z < depth_lower_threshold:
        #     drone.set_roll(0)
        #     drone.set_throttle(0)
        #     print('move backward')
        #     drone.set_pitch(-pitch_power)
        #     drone.move(1)
        else:
            drone.set_pitch(0)
            drone.set_throttle(0)
            drone.move(0)

def drone_land(drone, x,z,drone_start_timer):
    if depth_lower_threshold <= z <= depth_upper_threshold and left_threshold <= x <= right_threshold:
        drone.go_to_height(height_threshold)
        drone_end_timer = time.time()
        print('Total time taken from takeoff to landing:', drone_end_timer - drone_start_timer)

        print('coords before landing', x, z)
        # drone.set_pitch(5)
        # drone.move(1)
        print('landing')
        drone.land()
        print("landing")
        drone.close()
        print('drone closed')
        drone.disconnect()
        print('drone disconnected')
    else:
        drone_move(drone,x,z)

    # executionTime = (time.time() - startTime)
    # print('Execution time in seconds: ' + str(executionTime))

if __name__ == '__main__':
    freeze_support()

    data_flag = Value('i',0)
    locationQ = Queue()

    detected_flag = Value('i',0)

    # camera = Process(target=camera_thread_function, args=(locationQ,data_flag))
    camera = Process(target=camera_thread_function, args=(locationQ,detected_flag,data_flag))
    camera.start()

    # drone1 = Process(target=drone_thread_function2, args=(locationQ,data_flag))
    drone1 = Process(target=drone_thread_function2, args=(locationQ,detected_flag,data_flag))
    drone1.start()

    camera.join()
    drone1.join()


