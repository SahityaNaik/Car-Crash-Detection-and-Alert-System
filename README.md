## Car Crash Detection and Alert System

This project is a Car Crash Detection and Alert System designed to detect vehicle collisions using video input. The system utilizes the YOLOv3 model for real-time object detection and provides alerts via speech synthesis and email notifications.

## Technologies Used

- Python  
- OpenCV (DNN module for YOLOv3)  
- Tkinter (for GUI)  
- pyttsx3 (text-to-speech)  
- smtplib (for sending email notifications)  
- python-dotenv (for secure credential management)

## Installation

1. **Clone the repository:**  
   ```bash
   git clone https://github.com/SahityaNaik/Car-Crash-Detection-and-Alert-System.git
   cd Car-Crash-Detection-and-Alert-System
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   ```
   - Windows: `.venv\Scripts\activate`
   - macOS/Linux: `source .venv/bin/activate`

3. **Install required packages:**  
   ```bash
   pip install -r requirements.txt
   ```
   Or manually:
   ```bash
   pip install opencv-python pyttsx3 numpy imutils python-dotenv
   ```

4. **Download the YOLOv3 weights file:**  
   - Download the weights file from Google Drive: [YOLOv3 Weights](https://drive.google.com/file/d/11wnDebtXz_LFNycm-I3trNsdJ96d-OUQ/view?usp=sharing)
   - Place the `yolov3.weights` file in the `yolo-coco/` directory of your project.

5. **Set up environment variables:**
   - Copy `env.example` to `.env`
   - Fill in your email credentials:
     ```
     CRASH_MAIL_USER=your_email@gmail.com
     CRASH_MAIL_PASS=your_app_password
     CRASH_MAIL_SENDER_NAME=Crash Alert Bot
     CRASH_MAIL_TO=recipient@example.com
     CRASH_MAIL_TO_NAMES=Recipient Name
     ```
   - **Note:** For Gmail, you'll need to use an [Application-Specific Password](https://myaccount.google.com/apppasswords) if 2FA is enabled.

## Usage  

1. Run the file: `python carcrashtkinter.py`
2. Click on the "DETECT FROM VIDEO" button to select a video file for processing.
3. The system will analyze the video and provide alerts if a crash is detected.

## How It Works  

The system uses the YOLOv3 object detection model to:
- Identify vehicles in a video stream.
- Detect potential crashes based on proximity and behavior of detected objects.
- Trigger alerts using voice synthesis and send an email notification with an image of the crash.  

## File Descriptions  

- `carcrashtkinter.py`: Main application file that includes the GUI and crash detection logic.  
- `sendmail.py`: Contains functions for sending email alerts (uses environment variables from `.env`).  
- `yolo-coco/yolov3.weights`: YOLOv3 model weights for object detection (download from Google Drive).  
- `yolo-coco/coco.names`: Class labels used by the YOLO model.  
- `yolo-coco/yolov3.cfg`: Configuration file for the YOLOv3 model.  
- `output/`: Directory where detected crash images are saved.  
- `Images/crash1.png`: Background image used in the GUI.
- `.env`: Environment variables file for email credentials 
- `env.example`: Template file showing required environment variables.
