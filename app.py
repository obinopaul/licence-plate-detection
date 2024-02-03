# Library imports
import numpy as np
import streamlit as st
import math 
import cv2
import cvzone 
import tempfile
from PIL import Image
from ultralytics import YOLO


# vid1 = r'data/vidcar.mp4'


# Load the YOLO model
model = YOLO('models/numberplatemodel.pt')
# classnames = ['LicensePlate']
classnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'LicensePlate', 'M', 'N', 'P', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z']


def process_image(image):
    # Convert the image to an array
    image_array = np.array(image)
    results = model(image_array)

    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = box.conf[0]
            class_detect = box.cls[0]
            class_detect = int(class_detect)
            class_detect = classnames[class_detect]
            conf = math.ceil(confidence * 100)
            if conf > 50 and class_detect == 'LicensePlate':
                cv2.rectangle(image_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cvzone.putTextRect(image_array, f'{class_detect}', [x1 + 8, y1 - 12], thickness=2, scale=1)
                # cv2.putText(image_array, f'{class_detect}', (x1 + 8, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return image_array

def main():
    st.title("License Plate Detection")
    file = st.file_uploader("Upload a video or image", type=["mp4", "jpg", "png"])

    if file is not None:
        if file.type == "video/mp4":
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(file.read())
            cap = cv2.VideoCapture(tfile.name)

            stframe = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = process_image(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame, channels="RGB")

            cap.release()
            cap.destroyAllWindows() 
        else:
            image = Image.open(file)
            image_array = process_image(image)
            st.image(image_array, use_column_width=True)

if __name__ == '__main__':
    main()
