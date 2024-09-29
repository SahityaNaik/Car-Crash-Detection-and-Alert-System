import tkinter as tk
from tkinter import filedialog
import sys
import cv2
import numpy as np
import pyttsx3
import sendmail
import time
engine = pyttsx3.init()
engine.setProperty("rate", 120)



def detect_from_video():
    # Open file dialog to select video file
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
    # Update label to display selected file path
    if file_path:
        selected_video_label.config(text=file_path)
    # load the COCO class labels our YOLO model was trained on
        #labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
        labelsPath = 'yolo-coco/coco.names'
        LABELS = open(labelsPath).read().strip().split("\n")  #Changes here

        # initialize a list of colors to represent each possible class label
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                dtype="uint8")

        #weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
        #configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])
        weightsPath = 'yolo-coco/yolov3.weights'
        configPath = 'yolo-coco/yolov3.cfg'

        print("[INFO] loading YOLO from disk...")
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        ln = net.getLayerNames()   #Changes here
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

        # initialize the video stream, pointer to output video file, and
        # frame dimensions
        print(file_path)
        vs = cv2.VideoCapture(file_path)
        #vs = cv2.VideoCapture(0)
        writer = None
        (W, H) = (None, None)

        try:
                prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
                        else cv2.CAP_PROP_FRAME_COUNT
                total = int(vs.get(prop))
                print("[INFO] {} total frames in video".format(total))

        except:
                print("[INFO] could not determine # of frames in video")
                print("[INFO] no approx. completion time can be provided")
                total = -1
        cnt = 0 
        fno = 0
        # ------------------FRAME PART-----------------------------------------
        counter = 0
        while True:
                start1 = time.time()
                (grabbed, frame) = vs.read()
                if(cnt%2!=0):
                        cnt+=1
                        continue
                fno+=1
                print("Frame No:", fno)
                if not grabbed:
                        break
                # if the frame dimensions are empty, grab them
                if W is None or H is None:
                        (H, W) = frame.shape[:2]

                blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                        swapRB=True, crop=False)
                net.setInput(blob)
                start = time.time()
                layerOutputs = net.forward(ln)
                end = time.time()

                boxes = []
                confidences = []
                classIDs = []
                # loop over each of the layer outputs
                for output in layerOutputs:
                        # loop over each of the detections
                        for detection in output:
                                # extract the class ID and confidence (i.e., probability)
                                # of the current object detection
                                scores = detection[5:]
                                classID = np.argmax(scores)
                                confidence = scores[classID]
                                # filter out weak predictions by ensuring the detected
                                # probability is greater than the minimum probability
                                if confidence > 0.5:
                                    #print("hello")
                                    box = detection[0:4] * np.array([W, H, W, H])
                                    (centerX, centerY, width, height) = box.astype("int")

                                    x = int(centerX - (width / 2))
                                    y = int(centerY - (height / 2))

                                    # update our list of bounding box coordinates,
                                    # confidences, and class IDs
                                    boxes.append([x, y, int(width), int(height)])
                                    confidences.append(float(confidence))
                                    classIDs.append(classID)
                                    

                # apply non-maxima suppression to suppress weak, overlapping
                # bounding boxes
                idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,
                        0.3)

                # ensure at least one detection exists
                if len(idxs) > 0:
                        idArray = []
                        for j in idxs.flatten():
                                if classIDs[j]==2 or classIDs[j]==0:
                                        idArray.append(j)
                        flag = 0			
                        for j in idArray:
                                for k in idArray:
                                        if k==j:
                                                continue	
                                        (x1, y1) = (boxes[j][0], boxes[j][1])	
                                        (w1, h1) = (boxes[j][2], boxes[j][3])
                                        (x2, y2) = (x1+w1, abs(y1-h1))
                                        
                                        (x3, y3) = (boxes[k][0], boxes[k][1])	
                                        (w2, h2) = (boxes[k][2], boxes[k][3])
                                        (x4, y4) = (x3+w2, abs(y3-h2))
                                        if(x1>x4 or x3>x2):
                                                flag = 1
                                                counter = 0
                                        if(y1 < y4 or y3 < y2):
                                                flag = 1
                                                counter = 0
                                        if flag == 0:
                                                counter+=1
                                                break
                                if flag == 0:
                                        break
                        # loop over the indexes we are keeping
                        if(counter==8):
                                engine = pyttsx3.init()
                                engine.say('crash detected')
                                engine.say('crash detected')
                                engine.say('crash detected')
                                engine.runAndWait()
                                print("CRASH ALERT")
                                cv2.imwrite("output/Crash.jpg", frame)
                                cv2.imshow('Crash detected',frame)
                                cv2.waitKey(0)
                                cv2.destroyAllWindows()
                                counter=0
                                sendmail.sendalert()
                        for i in idxs.flatten():
                                # extract the bounding box coordinates
                                (x, y) = (boxes[i][0], boxes[i][1])
                                (w, h) = (boxes[i][2], boxes[i][3])

                                # draw a bounding box rectangle and label on the frame
                                color = [int(c) for c in COLORS[classIDs[i]]]
                                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                                text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                                        confidences[i])
                                cv2.putText(frame, text, (x, y - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                print(f"Box[{i}]: {x} {y} {w} {h} Labels[{i}]: {LABELS[classIDs[i]]} classIDs[{i}]: {classIDs[i]}  confidences[{i}]: {confidences[i]}")			
                end1 = time.time()
                #cv2.imwrite("output/frame{}.jpg".format(fno), frame)
                print("Complete time for algorithm", (end1-start1))
                cnt+=1
        print("[INFO] cleaning up...")
        
        vs.release()

# Create the main window
root = tk.Tk()
root.title("CAR CRASH DETECTION AND ALERT SYSTEM")

# Set window size and center it on screen
window_width = 600
window_height = 400
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2
root.geometry(f"{window_width}x{window_height}+{x}+{y}")

# Load background image
background_image = tk.PhotoImage(file="crash1.png")

# Create a label with the background image
background_label = tk.Label(root, image=background_image)
background_label.place(relwidth=1, relheight=1)

# Create header label
header_label = tk.Label(root, text="CAR CRASH DETECTION AND ALERT SYSTEM", font=("Arial", 20, "bold"))
header_label.place(relx=0.5, rely=0.1, anchor="center")

# Create buttons
'''detect_live_button = tk.Button(root, text="DETECT FROM LIVE CAMERA", command=detect_live_camera)
detect_live_button.place(relx=0.5, rely=0.4, anchor="center")'''

detect_video_button = tk.Button(root, text="DETECT FROM VIDEO", command=detect_from_video)
detect_video_button.place(relx=0.5, rely=0.6, anchor="center")

# Create label to display selected video file path
selected_video_label = tk.Label(root, text="", bg="white", wraplength=400)
selected_video_label.place(relx=0.5, rely=0.7, anchor="center")

root.mainloop()
