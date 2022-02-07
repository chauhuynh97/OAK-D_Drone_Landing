import cv2
import numpy as np
from pyzbar.pyzbar import decode

video_capture = cv2.VideoCapture(0)
video_capture.set(3, 1920)
video_capture.set(4, 1080)

# Need to define a target item to detect
item_to_detect = "coDrone"
keyVal = 0

if video_capture.isOpened():
    while keyVal != 33:
        # openCV reads frame (in this case, from live stream) and returns a success result and image
        success, img = video_capture.read()
        
        # A frame can have multiple qrcodes, we need to access each of them and process them.
        for qrcode_inFrame in decode(img):
            # Testing purposes
            print(qrcode_inFrame)
            print(qrcode_inFrame.data)
            print(qrcode_inFrame.rect)

            # Data is returned in bytes, we need to convert it into the desired format: utf-8 string type
            QRCode_data = qrcode_inFrame.data.decode('utf-8')
            print(QRCode_data)

            if QRCode_data == item_to_detect:
                polygon_conversion = np.array([qrcode_inFrame.polygon], np.int32)
                polygon_conversion = polygon_conversion.reshape((-1, 1, 2))
                print(polygon_conversion)
                top_left_x = qrcode_inFrame.rect[0]
                top_left_y = qrcode_inFrame.rect[1]

                bot_right_x = top_left_x + qrcode_inFrame.rect[2]
                bot_right_y = top_left_y + qrcode_inFrame.rect[3]

                pt1 = [top_left_x, top_left_y]
                pt2 = [bot_right_x, bot_right_y]

                # For testing purposes
                print(pt1)
                print(pt2)

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
