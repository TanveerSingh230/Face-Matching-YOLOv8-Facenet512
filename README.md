# Face-Matching-YOLOv8-Facenet512
This project is a web application that verifies if two faces match or not. Users can input two image URLs through an HTML interface, and the application processes the images to determine if the faces in the images match. The application is built using FastAPI and is deployed on a web server.

## Features

- **Face Detection and Verification**: Utilizes YOLOv8 for face detection and DeepFace with FaceNet512 for face verification.
- **Web Interface**: An HTML interface allows users to enter two image URLs and get the verification result.
- **FastAPI Backend**: A FastAPI backend processes the images and performs the face verification.
- **Automatic Orientation Correction**: The application corrects the orientation of the images to maximize detection accuracy.

## Usage

1. Open the web interface in your browser.
2. Enter the URLs of the two images you want to compare.
3. Click the "Compare" button.
4. The application will process the images and display the result: "Faces match" or "Faces do not match".

## FaceNet and DeepFace
- **FaceNet**:
        FaceNet is a deep learning model developed by Google that achieves state-of-the-art accuracy in face recognition tasks.
        It maps faces into a 128-dimensional embedding space where the distance between two embeddings corresponds to the similarity of the faces.
        In this project, FaceNet512 (a version of FaceNet with 512 dimensions) is used for extracting robust face embeddings.
- **DeepFace**:
        DeepFace is an open-source Python library for deep learning-based face recognition, facial attribute analysis, and face verification.
        It provides a high-level interface to use various state-of-the-art models, including FaceNet, for face recognition tasks.

## Code Explanation

Functions:

    load_image_from_url(url): Fetches and decodes the image from the provided URL.
    detect_faces_yolo(image): Detects faces in the image using YOLOv8.
    rotate_image(image, angle): Rotates the image by the given angle.
    detect_faces_with_rotation(image): Detects faces with rotation contingency.
    get_face_embeddings(image, bboxes): Extracts face embeddings using FaceNet512 via DeepFace.
    compare_faces(embedding1, embedding2): Compares two face embeddings using cosine similarity.
    compare_images_from_urls(url1, url2): Main function to compare two images from URLs.

### `main.py`

This is the main FastAPI application file.

- **Dependencies**:
  - `cv2`: OpenCV library for image processing.
  - `requests`: To fetch images from the provided URL.
  - `fastapi`: The FastAPI framework.
  - `ultralytics`: YOLOv8 model for object detection.
  - `deepface`: DeepFace library for face recognition.
  - `numpy`: For numerical operations.

-**Link for YOLOv8 Face Model**
https://github.com/akanametov/yolo-face/tree/dev

- **Load the YOLOv8 model**:
  ```python
  yolo_model = YOLO('yolov8l-face.pt')  # Ensure you have the appropriate YOLOv8 model weights
