
import cv2
import depthai as dai
import numpy as np
import blobconverter
import CoDrone
from multiprocessing import Process, Manager, Value, Queue, freeze_support
import time
from pyzbar.pyzbar import decode
from ctypes import c_char_p

height_threshold = 200  # height before drone.land()
depth_upper_threshold = 1000
depth_lower_threshold = 300
left_threshold = -300
right_threshold = 300

drone_movement_start = 0
drone_movement_end = 0
data_request_start = 0
data_request_end = 0

yaw_power = 20
pitch_power = 15
roll_power = 10


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


def drone_thread_function2(z_location,Qrcode_value):
    # set up drone connection and take off before trying to land
    print('Creating Drone Object')
    drone = CoDrone.CoDrone()
    print("Getting Ready to Pair")
    # drone.pair(drone.Nearest)
    # drone.pair('1484','COM6')
    drone.pair()
    print("Paired")
    # data_flag.value = 1
    # drone.takeoff()
    # drone.set_throttle(50)
    # drone_start_timer = time.time()
    # print("Taking Off")
    # print(drone.get_height())
    # drone.go_to_height(400)
    #     "Rotate Right", "Rotate Left", "Rotate End", "Move Forward", "Move Right", "Move Left", "Move End"

    #drone.set_pitch(20)

    while True:
        if Qrcode_value.value == "Takeoff":
            drone.takeoff()
            print("Taking Off")
        elif Qrcode_value.value == "Rotate Right":
            drone.set_yaw(yaw_power)
            while True:
                drone.move()
                if Qrcode_value.value != "Rotate Right":
                    break
        elif Qrcode_value.value == "Rotate Left":
            drone.set_yaw(-yaw_power)
            while True:
                drone.move()
                if Qrcode_value.value != "Rotate Left":
                    break
        elif Qrcode_value.value == "Move Forward":
            drone.set_pitch(pitch_power)
            while True:
                drone.move()
                # data_flag.value = 1
                if z_location.value < 1500 and z_location.value != 0:
                    # data_flag.value = 1
                    # print(q.get(0))
                    # start_coords = q.get(0)
                    # xs,ys,zs = start_coords
                    # print(xs,zs)
                    detected_land(drone,z_location)
                    # drone_land2(drone,q,detected_flag,data_flag)
                    break
            break
        elif Qrcode_value.value == "Move Right":
            drone.set_roll(roll_power)
            while True:
                drone.move()
                if Qrcode_value.value != "Move Right":
                    break
        elif Qrcode_value.value == "Move Left":
            drone.set_roll(-roll_power)
            while True:
                drone.move()
                if Qrcode_value.value != "Move Left":
                    break
        elif Qrcode_value.value == "Hover":
            drone.set_yaw(0)
            drone.set_pitch(0)
            drone.set_roll(0)
            while True:
                drone.move()
                if Qrcode_value.value != "Hover":
                    break
        drone.set_yaw(0)
        drone.set_pitch(0)
        drone.set_roll(0)


def detected_land(drone,z_location):
    # if detected_flag.value == 1 and not q.empty():
    #     data_flag.value = 1
    #     start_coords2 = q.get(0)
    #     xl,yl,zl = [0,0,0]
    #     xs,ys,zs = start_coords2
    #     x = xs - xl
    #     z = zs - zl
    #
    #     print(x,z)

    print("Drone detected by camera. Initiate landing ...")
    drone.go_to_height(height_threshold)

    drone.set_pitch(10)
    drone.move(1)

    drone.land()
    print("landing")
    drone.close()
    print('drone closed')
    drone.disconnect()
    print('drone disconnected')


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
                drone.set_pitch(pitch_power)
                drone.move(4)

            if depth_lower_threshold <= z <= depth_upper_threshold and left_threshold <= x <= right_threshold:
                drone.go_to_height(height_threshold)
                drone.land()
                drone.close()
                print('drone closed')
                drone.disconnect()
                print('drone disconnected')
                break
            else:
                drone_move(drone,x,z)


def camera_QRcode_thread_function(Qrcode_value):
    video_capture = cv2.VideoCapture(0)
    video_capture.set(3, 1920)
    video_capture.set(4, 1080)

    # Need to define a target item to detect
    item_to_detect = ["Takeoff", "Rotate Right", "Rotate Left", "Move Forward", "Move Right", "Move Left", "Hover"]
    keyVal = 0

    if video_capture.isOpened():
        while keyVal != 33:
            # openCV reads frame (in this case, from live stream) and returns a success result and image
            success, img = video_capture.read()

            # A frame can have multiple qrcodes, we need to access each of them and process them.
            for qrcode_inFrame in decode(img):

                # Data is returned in bytes, we need to convert it into the desired format: utf-8 string type
                QRCode_data = qrcode_inFrame.data.decode('utf-8')

                if QRCode_data in item_to_detect:

                    Qrcode_value.value = QRCode_data

                    polygon_conversion = np.array([qrcode_inFrame.polygon], np.int32)
                    polygon_conversion = polygon_conversion.reshape((-1, 1, 2))

                    # print(polygon_conversion)

                    top_left_x = qrcode_inFrame.rect[0]
                    top_left_y = qrcode_inFrame.rect[1]

                    bot_right_x = top_left_x + qrcode_inFrame.rect[2]
                    bot_right_y = top_left_y + qrcode_inFrame.rect[3]

                    pt1 = [top_left_x, top_left_y]
                    pt2 = [bot_right_x, bot_right_y]

                    # For testing purposes
                    # print(pt1)
                    # print(pt2)

                    # Only draw bounding boxes if the QRCode matches the QR_code
                    cv2.polylines(img, [polygon_conversion], True, (0, 0, 255), 4)

                    # For testing purposes
                    polygon_text_placement = qrcode_inFrame.rect
                    cv2.putText(img, QRCode_data, (polygon_text_placement[0], polygon_text_placement[1]),
                                cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 0, 0), 2)

            cv2.imshow('Result', img)
            keyVal = cv2.waitKey(33)
            if keyVal == ord('a'):
                print("pressed A - Exit request detected!!")
    else:
        print("Camera turned off, Goodbye.")

    video_capture.release()
    exit()


# def camera_thread_function(q,data_flag):
def camera_thread_function(z_location):

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
                    z_location.value = zs
                    # detected_flag.value = 1
                    # if data_flag.value == 1:
                    #     # q.put([xs,0,zs])
                    #
                    #     data_flag.value = 0
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
        # drone.move(0)
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
    # locationQ = Queue()
    z_location = Value('i',30000)
    manager = Manager()
    Qrcode_value = manager.Value(c_char_p, "")

    detected_flag = Value('i',0)

    # camera = Process(target=camera_thread_function, args=(locationQ,data_flag))
    camera = Process(target=camera_thread_function, args=(z_location,))
    camera.start()

    # QR webcam
    webcam = Process(target=camera_QRcode_thread_function, args=(Qrcode_value,))
    webcam.start()

    # drone1 = Process(target=drone_thread_function2, args=(locationQ,data_flag))
    drone1 = Process(target=drone_thread_function2, args=(z_location,Qrcode_value))
    drone1.start()

    camera.join()
    webcam.join()
    drone1.join()


