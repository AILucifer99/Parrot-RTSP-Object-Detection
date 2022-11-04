import cv2
import numpy as np
import os
import random
import time
import math
import argparse


class parrotRTSPObjectDetection :
    def __init__(self, rtspLink) :
        super(parrotRTSPObjectDetection, self).__init__()
        self.rtspLink = rtspLink

    def SSD_RTSP_OBJECT_DETECTION(self) :

        def parseIndex(val1, val2, control, parse) :
            if parse :
                x = random.choice([random.uniform(val1, val2) for _ in range(control)])
                return math.ceil(x)

        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
        
        #Parameters for tuning the detected video and streaming 
        OUTPUT_FILENAME = f'./Results/Sample_{parseIndex(1, 50, 5, True)}_{time.time}_.mp4' # We want to save the output to a video file
        FRAMES_PER_SECOND = 15.0
        RESIZED_DIMENSIONS = (300, 300) # Dimensions that SSD was trained on. 
        IMG_NORM_RATIO = 0.007843 # In grayscale a pixel can range between 0 and 255
        FILE_SIZE = (640, 680) # Assumes 1920x1080 mp4

        # Load the pre-trained neural network
        neural_network = cv2.dnn.readNetFromCaffe(
            './Resources/MobileNetSSD_txt.prototxt', 
            './Resources/MobileNetSSD_model.caffemodel'
            )
        
        # List of categories and classes
        categories = { 0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 
                    4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 
                    9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog', 
                    13: 'horse', 14: 'motorbike', 15: 'person', 
                    16: 'pottedplant', 17: 'sheep', 18: 'sofa', 
                    19: 'train', 20: 'tvmonitor'}
        
        classes =  ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", 
                    "bus", "car", "cat", "chair", "cow", 
                "diningtable",  "dog", "horse", "motorbike", "person", 
                "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


        # Create the bounding boxes
        bbox_colors = np.random.uniform(255, 0, size=(len(categories), 3))

        class ParrotObjectDetectorDrone :
            def __init__(self, rtspLink, detectionConf, resizingConf, detectObject) :
                super(ParrotObjectDetectorDrone, self).__init__()
                self.rtspLink = rtspLink
                self.detectionConf = detectionConf
                self.resizingConf = resizingConf
                self.detectObject = detectObject


            def PerformObjectDetection(self) :
                if self.detectObject :
                    print("Code Running.....")
                    vcap = cv2.VideoCapture(self.rtspLink, cv2.CAP_FFMPEG)
                    
                    # Create a VideoWriter object so we can save the video output
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    result = cv2.VideoWriter(OUTPUT_FILENAME,  
                                            fourcc, 
                                            FRAMES_PER_SECOND, 
                                            FILE_SIZE) 
                    while True:
                        ret, frame = vcap.read()
                        if ret == False:
                            print("Frame is empty")
                            break
                        
                        else:
                            # Capture the frame's height and width
                            (h, w) = frame.shape[:2]
                        
                            # Create a blob. A blob is a group of connected pixels in a binary 
                            # frame that share some common property (e.g. grayscale value)
                            # Preprocess the frame to prepare it for deep learning classification
                            frame_blob = cv2.dnn.blobFromImage(
                                cv2.resize(frame, RESIZED_DIMENSIONS), 
                                IMG_NORM_RATIO, RESIZED_DIMENSIONS, self.resizingConf, 
                                )
                            
                            # Set the input for the neural network
                            neural_network.setInput(frame_blob)
                        
                            # Predict the objects in the image
                            neural_network_output = neural_network.forward()
                            
                            # Put the bounding boxes around the detected objects
                            for i in np.arange(0, neural_network_output.shape[2]):
                                confidence = neural_network_output[0, 0, i, 2]
                                
                                # Confidence must be at greater than 25.75 percent    
                                if confidence > self.detectionConf :
                                    idx = int(neural_network_output[0, 0, i, 1])
                                    bounding_box = neural_network_output[0, 0, i, 3:7] * np.array([w, h, w, h])
                                    (startX, startY, endX, endY) = bounding_box.astype("int")
                                    label = "{}: {:.2f}%".format(classes[idx], confidence * 100) 
                        
                                    cv2.rectangle(frame, (startX, startY), (endX, endY), bbox_colors[idx], 3)     
                                        
                                    y = startY - 15 if startY - 15 > 15 else startY + 15    
                
                                    cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, bbox_colors[idx], 3)

                            # We now need to resize the frame so its dimensions
                            # are equivalent to the dimensions of the original frame
                            frame = cv2.resize(frame, FILE_SIZE, interpolation=cv2.INTER_NEAREST)
                            # Write the frame to the output video file
                            result.write(frame)
                            cv2.imshow('VIDEO', frame)
                            
                            if cv2.waitKey(1) & 0xFF == ord('s') :
                                print("Ending...")
                                break
                    
                    # Stop when the video is finished
                    vcap.release()  
                    # Release the video recording
                    result.release()  


        if __name__ == "__main__" :
            rtsp_link = self.rtspLink
            detection_confidence = 0.35
            resizing_confidence = 127.5
            
            drone = ParrotObjectDetectorDrone(rtsp_link, detection_confidence, resizing_confidence, True)
            drone.PerformObjectDetection()


    def YOLO_RTSP_OBJECT_DETECTION(self) :

        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

        parser = argparse.ArgumentParser()
        parser.add_argument('--rtsp', help="True/False", default=True)
        parser.add_argument('--play_video', help="Tue/False", default=False)
        parser.add_argument('--image', help="Tue/False", default=False)
        parser.add_argument('--video_path', help="Path of video file", default="videos/car_on_road.mp4")
        parser.add_argument('--image_path', help="Path of image to detect objects", default="Images/bicycle.jpg")
        parser.add_argument('--verbose', help="To print statements", default=True)
        args = parser.parse_args()
        time.sleep(1)


        def load_yolo_model():
            net = cv2.dnn.readNet("./src/yolo-files/yolov3.weights", "./src/yolo-files/yolov3.cfg")
            classes = []
            with open("./src/yolo-files/coco.names", "r") as f:
                classes = [line.strip() for line in f.readlines()] 
            
            output_layers = [layer_name for layer_name in net.getUnconnectedOutLayersNames()]
            colors = np.random.uniform(0, 255, size=(len(classes), 3))
            return net, classes, colors, output_layers


        def load_image(img_path):
            # image loading
            img = cv2.imread(img_path)
            img = cv2.resize(img, None, fx=0.4, fy=0.4)
            height, width, channels = img.shape
            return img, height, width, channels

        def start_rtsp_video_link():
            cap = cv2.VideoCapture(self.rtspLink)
            return cap


        def display_blob(blob):
            for b in blob:
                for n, imgb in enumerate(b):
                    cv2.imshow(str(n), imgb)


        def detect_objects(img, net, outputLayers):			
            blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, 
                                            size=(320, 320), 
                                            mean=(0, 0, 0), 
                                            swapRB=True, crop=False)
            net.setInput(blob)
            outputs = net.forward(outputLayers)
            return blob, outputs


        def get_box_dimensions(outputs, height, width):
            boxes = []
            confs = []
            class_ids = []
            for output in outputs:
                for detect in output:
                    scores = detect[5:]
                    class_id = np.argmax(scores)
                    conf = scores[class_id]
                    if conf > 0.3:
                        center_x = int(detect[0] * width)
                        center_y = int(detect[1] * height)
                        w = int(detect[2] * width)
                        h = int(detect[3] * height)
                        x = int(center_x - w/2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confs.append(float(conf))
                        class_ids.append(class_id)
            return boxes, confs, class_ids


        def draw_labels(boxes, confs, colors, class_ids, classes, img): 
            indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
            font = cv2.FONT_HERSHEY_PLAIN
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    color = colors[i]
                    cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                    cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
            cv2.imshow("Image", img)


        def image_detect(img_path): 
            model, classes, colors, output_layers = load_yolo_model()
            image, height, width, channels = load_image(img_path)
            blob, outputs = detect_objects(image, model, output_layers)
            boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
            draw_labels(boxes, confs, colors, class_ids, classes, image)
            while True:
                key = cv2.waitKey(1)
                if key == 27:
                    break


        def rtsp_video_link_detect():
            model, classes, colors, output_layers = load_yolo_model()
            cap = start_rtsp_video_link()
            while True:
                _, frame = cap.read()
                height, width, channels = frame.shape
                blob, outputs = detect_objects(frame, model, output_layers)
                boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
                draw_labels(boxes, confs, colors, class_ids, classes, frame)
                key = cv2.waitKey(1)
                if key == 27:
                    break
            cap.release()

        def start_video_detection(video_path):
            model, classes, colors, output_layers = load_yolo_model()
            cap = cv2.VideoCapture(video_path)
            while True:
                _, frame = cap.read()
                height, width, channels = frame.shape
                blob, outputs = detect_objects(frame, model, output_layers)
                boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
                draw_labels(boxes, confs, colors, class_ids, classes, frame)
                key = cv2.waitKey(1)
                if key == 27:
                    break
            cap.release()


        if __name__ == '__main__':
            rtsp = args.rtsp
            video_play = args.play_video
            image = args.image
            if rtsp:
                if args.verbose:
                    print('---- Starting Drone RTSP Link object detection ----')
                rtsp_video_link_detect()
            if video_play:
                video_path = args.video_path
                if args.verbose:
                    print('Opening '+video_path+" .... ")
                start_video_detection(video_path)
            if image:
                image_path = args.image_path
                if args.verbose:
                    print("Opening "+image_path+" .... ")
                image_detect(image_path)
            
            cv2.destroyAllWindows()


if __name__ == "__main__" :
    rtspUrl = "rtsp://192.168.53.1/live"
    drone = parrotRTSPObjectDetection(rtspLink=rtspUrl)
    drone.SSD_RTSP_OBJECT_DETECTION()
   # drone.YOLO_RTSP_OBJECT_DETECTION()
    

