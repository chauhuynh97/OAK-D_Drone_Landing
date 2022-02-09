import cv2
import numpy as np
import time
from threading import Thread
import queue
from pyzbar.pyzbar import decode
import sys
import depthai as dai
import blobconverter


window_name = 'QR Code Detection'
threshold_for_approval = 0.75

# Need to load class names and YOLOv3-tiny model to attain the lightweight
# framework to build upon. Darknet is a framework that uses CUDA, makes processing
# much faster. YOLOV3 also provides us with weights for QRcode, which is needed by the
# Neural network.
classes = open('qrcode.names').read().strip().split('\n')
net = cv2.dnn.readNetFromDarknet('qrcode-yolov3-tiny.cfg',
                                 'qrcode-yolov3-tiny_last.weights')
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

# Determine the output layer
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

pipeline = dai.Pipeline()

pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_2)

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

#
# Postprocessing and rendering loop
#
device = dai.Device(pipeline, usb2Mode=True)

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
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    classIds = []
    confidences = []
    boxes = []

    for detection in detections:
        # Denormalize bounding box
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
        x1 = int(detection.xmin * frameWidth)
        x2 = int(detection.xmax * frameWidth)
        y1 = int(detection.ymin * frameHeight)
        y2 = int(detection.ymax * frameHeight)

        cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
        cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
        cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

        # cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)
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

    cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
    cv2.imshow("depth", depthFrameColor)
    cv2.imshow("rgb", frame)

    if cv2.waitKey(1) == ord('q'):
        break
