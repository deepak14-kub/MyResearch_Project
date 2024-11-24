# Parking Lot System

This project demonstrates a Parking Lot System using YOLO for object detection and EasyOCR for number plate recognition.

## Features

- **Parking Lot Detection**: Detects parking spaces in an image of a parking lot.
- **Number Plate Detection**: Detects and reads number plates from an image of a car.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/parking-lot-system.git
    cd parking-lot-system
    ```

2. **Create a virtual environment**:
    ```bash
    python -m venv myenv
    source myenv/bin/activate  # On Windows, use `myenv\Scripts\activate`
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Download and set up Tesseract OCR**:
    - **Windows**: [Download and install Tesseract](https://github.com/UB-Mannheim/tesseract/wiki). Ensure it's added to your system's PATH.
    - **macOS**: Install Tesseract using Homebrew:
      ```bash
      brew install tesseract
      ```
    - **Linux**: Install Tesseract using apt-get:
      ```bash
      sudo apt-get install tesseract-ocr
      ```

5. **Download YOLO models**:
    - Place your YOLO models (`parking_yolov9_model.pt` and `anpr_model.pt`) in the project directory.

## Usage

1. **Run the Streamlit application**:
    ```bash
    streamlit run app.py
    ```

2. **Open your browser**: Navigate to `http://localhost:8501` to access the application.

3. **Select Task**: Choose either `Parking Lot Detection` or `Number Plate Detection`.

4. **Upload Image**: Upload an appropriate image based on the selected task.

5. **View Results**: The application will process the image and display the detection results.

## Contributing

Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.