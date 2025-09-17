import cv2
import os
import streamlit as st
import threading
import time
from ultralytics import YOLO
from pushbullet import Pushbullet

# -------------------- Pushbullet Setup --------------------
API_KEY = "o.2LDpOlCHQhc3Pe7e4v5zJ9BdiImKvJjj"  # Replace with your Pushbullet API key
pb = Pushbullet(API_KEY)

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Human Detection Security App", layout="wide")
st.title("🔐 Security Assistant")

run_detection = st.checkbox("Start Human Detection")

# -------------------- Load YOLOv8 Model --------------------
model = YOLO("yolov8n.pt")  # Ensure yolov8n.pt is in the same folder

# -------------------- Create folder to save detected frames --------------------
if not os.path.exists("human_frames"):
    os.makedirs("human_frames")

# -------------------- Streamlit video display --------------------
frame_display = st.empty()

# -------------------- Notification function (runs in background) --------------------
def send_alert(frame_path):
    try:
        pb.push_note("Human Detected!", "A human was detected by your YOLOv8 camera.")
        with open(frame_path, "rb") as pic:
            file_data = pb.upload_file(pic, "frame.jpg")
        pb.push_file(**file_data, title="Human Detected!")
    except Exception as e:
        print("Pushbullet Error:", e)

# -------------------- Human Detection Logic --------------------
if run_detection:
    cap = cv2.VideoCapture(0)  # Open webcam
    frame_count = 0
    last_sent = 0
    cooldown = 5  # seconds

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Could not read from webcam.")
            break

        # Resize frame for speed
        frame_resized = cv2.resize(frame, (640, 480))
        results = model(frame_resized)
        human_detected = False

        # Process YOLO results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if int(box.cls[0]) == 0:  # Class 0 = person
                    human_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_resized, 'Human', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Save frames and send Pushbullet notification (with cooldown)
        if human_detected and time.time() - last_sent > cooldown:
            frame_path = f"human_frames/frame_{frame_count}.jpg"
            cv2.imwrite(frame_path, frame_resized)
            frame_count += 1
            last_sent = time.time()

            # Run notification in background thread
            threading.Thread(target=send_alert, args=(frame_path,)).start()

        # Convert frame to RGB for Streamlit display
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_display.image(frame_rgb, channels="RGB")

        # Exit if checkbox unchecked
        if not st.session_state.get("Start Human Detection", run_detection):
            break

    cap.release()
    cv2.destroyAllWindows()

else:
    st.info("Check the box to start real-time detection.")
