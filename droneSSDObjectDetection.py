import cv2
import numpy as np
import os
import random
import time
import math


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
    rtsp_link = "rtsp://192.168.53.1/live"
    detection_confidence = 0.35
    resizing_confidence = 127.5
    
    drone = ParrotObjectDetectorDrone(rtsp_link, detection_confidence, resizing_confidence, True)
    drone.PerformObjectDetection()

    

