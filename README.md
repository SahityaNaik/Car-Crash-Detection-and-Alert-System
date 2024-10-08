## Car Crash Detection and Alert System

This project is a Car Crash Detection and Alert System designed to identify potential vehicle collisions using video input. The system utilizes the YOLOv3 model for real-time object detection and provides alerts via speech synthesis and email notifications.

## Technologies Used

- Python  
- OpenCV  
- TensorFlow/Keras (for YOLOv3)  
- Tkinter (for GUI)  
- pyttsx3 (text-to-speech)  
- smtplib (for sending email notifications)  

## Installation

1. Clone the repository:  
    - `git clone https://github.com/SahityaNaik/Car-Crash-Detection-and-Alert-System`  
    - `cd your-repo-name`

2. Install required packages:  
    - `pip install opencv-python pyttsx3 numpy`

3. Download the YOLOv3 weights file:  
    - Download the weights file from Google Drive: [YOLOv3 Weights](https://drive.google.com/file/d/11wnDebtXz_LFNycm-I3trNsdJ96d-OUQ/view?usp=sharing)

4. Place the yolov3.weights file in the `yolo-coco/` directory of your project.

## Usage  

1. Run the file: `carcrashtikinter.py`
2. Click on the "DETECT FROM VIDEO" button to select a video file for processing.
3. The system will analyze the video and provide alerts if a crash is detected.

## How It Works  

The system uses the YOLOv3 object detection model to:
- Identify vehicles in a video stream.
- Detect potential crashes based on proximity and behavior of detected objects.
- Trigger alerts using voice synthesis and send an email notification with an image of the crash.  

## File Descriptions  

- `carcrashtikinter.py`: Main application file that includes the GUI and crash detection logic.  
- `sendmail.py`: Contains functions for sending email alerts.  
- `yolov3.weights`: YOLOv3 model weights for object detection (hosted on Google Drive).  
- `yolo-coco/coco.names`: Class labels used by the YOLO model.  
- `yolo-coco/yolov3.cfg`: Configuration file for the YOLOv3 model.  
- `output/`: Directory where detected crash images are saved.  
- `crash1.png`: Background image used in the GUI.

## Note
In the `sendmail.py` file, replace the placeholders for:
- `sender_email`: Your email address
- `sender_name`: Your name
- `password`: Your email password (use an app-specific password if needed)
- `receiver_emails`: The recipient's email address
- `receiver_names`: The recipient's name
