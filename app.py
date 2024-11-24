import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
from collections import Counter
import pandas as pd
import easyocr
import cv2
import pytesseract

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize EasyOCR reader (this will download the model files on first run)
def load_ocr():
    return easyocr.Reader(['en'])

# Function to perform OCR on a number plate image
def read_plate(plate_img, reader):
    preprocessed_img = preprocess_image(plate_img)
    results = reader.readtext(preprocessed_img)
    
    # Extract and combine text
    if results:
        text = ' '.join([result[1] for result in results])
        # Clean the text (remove spaces and unwanted characters)
        text = ''.join(e for e in text if e.isalnum())
        return text
    return ''

# Image preprocessing function
def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    return binary_image

# Function to perform YOLO inference and OCR
def detect_objects(image, model, reader, confidence=0.5):
    img_array = np.array(image)
    results = model(img_array)  # Run inference
    annotated_image = results[0].plot()  # Annotate image
    
    # Extract bounding boxes and class IDs
    detections = results[0].boxes.data.cpu().numpy()  # Get bounding boxes
    filtered_detections = [
        det for det in detections if det[4] > confidence  # Filter by confidence
    ]
    
    # Extract class IDs
    class_ids = [int(det[5]) for det in filtered_detections]  # Index 5 contains class ID
    
    # Process number plates if any detected
    plate_texts = []
    if filtered_detections:
        for det in filtered_detections:
            x1, y1, x2, y2 = map(int, det[:4])  # Get coordinates
            # Extract plate region from original image
            plate_region = img_array[y1:y2, x1:x2]
            if plate_region.size > 0:  # Check if region is valid
                plate_text = read_plate(plate_region, reader)
                if plate_text:  # Only add if text was detected
                    plate_texts.append(plate_text)
    
    return Image.fromarray(annotated_image), filtered_detections, class_ids, plate_texts

# Streamlit UI
st.title("Parking Lot System")
st.write("Choose a detection task and upload an image.")

# Initialize EasyOCR
reader = load_ocr()

# Task Selection
task = st.radio("Select Task", ["Parking Lot Detection", "Number Plate Detection"])

if task == "Parking Lot Detection":
    st.header("Upload Image of Parking Lot")
    
    # File upload
    uploaded_file = st.file_uploader("Choose an image of a parking lot...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        st.write("Detecting parking spaces...")
        
        # Load the parking lot detection model
        model_path = "parking_yolov9_model.pt"
        model = YOLO(model_path)
        
        # Perform detection
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5)
        output_image, detections, class_ids, _ = detect_objects(image, model, reader, confidence=confidence_threshold)
        
        # Display results
        st.image(output_image, caption="Detection Results", use_column_width=True)
        st.write("Detections:")
        st.write(detections)
        
        # Count class occurrences
        class_counts = Counter(class_ids)
        df = pd.DataFrame(class_counts.items(), columns=["Class ID", "Count"])
        st.table(df)

elif task == "Number Plate Detection":
    st.header("Upload Image of Car")
    
    # File upload
    uploaded_file = st.file_uploader("Choose an image of a car...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        st.write("Detecting number plates...")
        
        # Load the number plate detection model
        model_path = "anpr_model.pt"
        model = YOLO(model_path)
        
        # Perform detection
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5)
        output_image, detections, class_ids, plate_texts = detect_objects(image, model, reader, confidence=confidence_threshold)
        
        # Display results
        st.image(output_image, caption="Detection Results", use_column_width=True)
        
        # Display detected number plates
        if plate_texts:
            st.subheader("Detected Number Plates:")
            for idx, plate in enumerate(plate_texts, 1):
                st.write(f"Plate {idx}: {plate}")
        else:
            st.write("No readable number plates detected")
        
        st.write("Detections:")
        st.write(detections)
        
        # Count class occurrences
        class_counts = Counter(class_ids)
        df = pd.DataFrame(class_counts.items(), columns=["Class ID", "Count"])
        st.table(df)
