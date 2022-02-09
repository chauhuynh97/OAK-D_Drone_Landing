import cv2
import numpy as np
import time
from threading import Thread
import queue
from pyzbar.pyzbar import decode
import sys
import depthai as dai
import blobconverter
import tensorflow as tf


window_name = 'QR Code Detection'
threshold_for_approval = 0.75

# Need to load class names and YOLOv3-tiny model to attain the lightweight
# framework to build upon. Darknet is a framework that uses CUDA, makes processing
# much faster. YOLOV3 also provides us with weights for QRcode, which is needed by the
# Neural network.
classes = open('/Users/arqumuddin/Desktop/OAK-D_Drone_Landing/Insitu Project/qr_code_detection/qrcode.names').read().strip().split('\n')
net = cv2.dnn.readNetFromDarknet('/Users/arqumuddin/Desktop/OAK-D_Drone_Landing/Insitu Project/qr_code_detection/qrcode-yolov3-tiny.cfg', 
                                 '/Users/arqumuddin/Desktop/OAK-D_Drone_Landing/Insitu Project/qr_code_detection/qrcode-yolov3-tiny_last.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

# YOLOV3 documentation will ask you to use this, but this is only for processing
# Images.
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) # DNN_TARGET_OPENCL DNN_TARGET_CPU

def labelQR(frame):
    for qrcode_inFrame in decode(frame):
        # Testing purposes
        print(qrcode_inFrame)
        print(qrcode_inFrame.data)
        print(qrcode_inFrame.rect)

        # Data is returned in bytes, we need to convert it into the desired format: utf-8 string type
        QRCode_data = qrcode_inFrame.data.decode('utf-8')
        print(QRCode_data)

        if QRCode_data == 'coDrone':
            polygon_conversion = np.array([qrcode_inFrame.polygon], np.int32)
            polygon_conversion = polygon_conversion.reshape((-1, 1, 2))

            top_left_x = qrcode_inFrame.rect[0]
            top_left_y = qrcode_inFrame.rect[1]

            bot_right_x = top_left_x + qrcode_inFrame.rect[2]
            bot_right_y = top_left_y + qrcode_inFrame.rect[3]

            pt1 = [top_left_x, top_left_y]
            pt2 = [bot_right_x, bot_right_y]

            # For testing purposes
            print(pt1)
            print(pt2)

            polygon_text_placement = qrcode_inFrame.rect
            cv2.putText(frame, QRCode_data, (polygon_text_placement[0], polygon_text_placement[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > threshold_for_approval:
                x, y, width, height = detection[:4] * np.array([frameWidth, frameHeight, frameWidth, frameHeight])
                left = int(x - width / 2)
                top = int(y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, int(width), int(height)])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, threshold_for_approval, threshold_for_approval - 0.1)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]

        cv2.rectangle(frame, (left, top), (left+width, top+height), (0, 0, 255), 2)
        # Draw class name and confidence
        label = '%s:%.2f' % (classes[classIds[i]], confidences[i])
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    labelQR(frame)

# Determine the output layer
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# cap = cv2.VideoCapture(0)

class QueueFPS(queue.Queue):
    def __init__(self):
        queue.Queue.__init__(self)
        self.startTime = 0
        self.counter = 0

    def put(self, v):
        queue.Queue.put(self, v)
        self.counter += 1
        if self.counter == 1:
            self.startTime = time.time()

    def getFPS(self):
        return self.counter / (time.time() - self.startTime)


process = True

# Frames capturing thread
framesQueue = QueueFPS()
def framesThreadBody():
    global framesQueue, process

    while process:
        hasFrame, frame = cap.read()
        if not hasFrame:
            break
        framesQueue.put(frame)


#
# Frames processing thread
#
processedFramesQueue = queue.Queue()
predictionsQueue = QueueFPS()
# Create pipeline
pipeline = dai.Pipeline()

def processingThreadBody():
    global processedFramesQueue, predictionsQueue, process

    while process:
        # Get a next frame
        frame = None
        try:
            frame = framesQueue.get_nowait()
            framesQueue.queue.clear()
        except queue.Empty:
            pass

        if not frame is None:
            nnBlobPath = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), swapRB=True, crop=False)

            syncNN = True

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

            # spatialDetectionNetwork.setBlobPath(blobconverter.from_zoo(name='qrcode-yolov3-tiny', shaves=6))
            spatialDetectionNetwork.setConfidenceThreshold(0.5)
            spatialDetectionNetwork.input.setBlocking(False)
            spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
            spatialDetectionNetwork.setDepthLowerThreshold(100)
            spatialDetectionNetwork.setDepthUpperThreshold(5000)

            # Yolo specific parameters
            spatialDetectionNetwork.setNumClasses(80)
            spatialDetectionNetwork.setCoordinateSize(4)
            spatialDetectionNetwork.setAnchors(np.array([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319]))
            spatialDetectionNetwork.setAnchorMasks({"side26": np.array([1, 2, 3]), "side13": np.array([3, 4, 5])})
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

            # image = np.copy(frame)
            processedFramesQueue.put(frame)

            # Run a model
            net.setInput(nnBlobPath)
            # Compute
            outs = net.forward(ln)

            outs = postprocess(frame, outs)

            predictionsQueue.put(outs)


framesThread = Thread(target=framesThreadBody)
framesThread.start()

processingThread = Thread(target=processingThreadBody)
processingThread.start()

#
# Postprocessing and rendering loop
#
device = dai.Device(pipeline, usb2Mode=True)
outs = predictionsQueue.get_nowait()

previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
xoutBoundingBoxDepthMappingQueue = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=4, blocking=False)
depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

startTime = time.monotonic()
counter = 0
fps = 0
color = (255, 255, 255)

while True:
    inPreview = previewQueue.get()
    inDet = detectionNNQueue.get()
    depth = depthQueue.get()

    frame = inPreview.getCvFrame()
    depthFrame = depth.getFrame() # depthFrame values are in millimeters

    depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
    depthFrameColor = cv2.equalizeHist(depthFrameColor)
    depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

    counter += 1
    current_time = time.monotonic()
    if (current_time - startTime) > 1:
        fps = counter / (current_time - startTime)
        counter = 0
        startTime = current_time

    detections = inDet.detections
    if len(detections) != 0:
        boundingBoxMapping = xoutBoundingBoxDepthMappingQueue.get()
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

            cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)


    # If the frame is available, draw bounding boxes on it and show the frame\
    height = frame.shape[0]
    width = frame.shape[1]
    for detection in detections:
        # Denormalize bounding box
        x1 = int(detection.xmin * width)
        x2 = int(detection.xmax * width)
        y1 = int(detection.ymin * height)
        y2 = int(detection.ymax * height)

        cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
        cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
        cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

        # cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

    cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
    cv2.imshow("depth", depthFrameColor)
    cv2.imshow("rgb", frame)

    if cv2.waitKey(1) == ord('q'):
        break
# with dai.Device(pipeline) as device:
#     try:
#         # Request prediction first because they put after frames
#         outs = predictionsQueue.get_nowait()
#         frame = processedFramesQueue.get_nowait()
#
#         # Put efficiency information.
#         if predictionsQueue.counter > 1:
#             label = 'Camera: %.2f FPS' % (framesQueue.getFPS())
#             cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
#
#             label = 'Network: %.2f FPS' % (predictionsQueue.getFPS())
#             cv2.putText(frame, label, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
#
#             label = 'Skipped frames: %d' % (framesQueue.counter - predictionsQueue.counter)
#             cv2.putText(frame, label, (0, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
#
#         cv2.imshow(window_name, frame)
#     except queue.Empty:
#         pass

process = False
framesThread.join()
processingThread.join()