from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
import requests
from ultralytics import YOLO
from deepface import DeepFace

# Initialize YOLOv8 model for face detection
yolo_model = YOLO('yolov8l-face.pt')  # Ensure you have the appropriate YOLOv8 model weights

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Function to load image from URL
def load_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        img = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error loading image from {url}: {e}")
        return None

# Function to detect faces using YOLOv8
def detect_faces_yolo(image):
    results = yolo_model(image, imgsz=1280)
    faces = []
    for result in results:
        for box in result.boxes:
            confidence = float(box.conf)  # Convert tensor to float
            if confidence > 0.5:  # Confidence threshold
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                faces.append((x1, y1, x2, y2, confidence))
    return faces

# Function to rotate image
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

# Function to detect faces with rotation contingency
def detect_faces_with_rotation(image):
    angles = [0, -90, 90]  # Original, 90 degrees left, 90 degrees right
    max_confidence = 0
    best_angle = 0
    best_faces = []
    best_image = None

    for angle in angles:
        rotated_image = rotate_image(image, angle)
        faces = detect_faces_yolo(rotated_image)
        if faces:
            highest_face_confidence = max(faces, key=lambda item: item[4])[4]
            if highest_face_confidence > max_confidence:
                max_confidence = highest_face_confidence
                best_angle = angle
                best_faces = faces
                best_image = rotated_image
            print(f"Faces detected at {angle} degrees rotation: {len(faces)} faces with max confidence {highest_face_confidence:.2f}")
    
    return best_image, best_faces

# Function to extract face embeddings using FaceNet512
def get_face_embeddings(image, bboxes):
    faces = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox[:4]  # Ignore confidence in bbox
        face = image[y1:y2, x1:x2]
        if face.size != 0:
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (160, 160))
            faces.append(face_resized)
    embeddings = []
    for face in faces:
        try:
            embedding = DeepFace.represent(face, model_name='Facenet512', enforce_detection=False)[0]['embedding']
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error encoding face: {e}")
    return embeddings

# Function to compare two face embeddings using cosine similarity
def compare_faces(embedding1, embedding2):
    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return similarity

# Main function to compare two images from URLs
def compare_images_from_urls(url1, url2):
    try:
        # Load images from URLs
        image1 = load_image_from_url(url1)
        image2 = load_image_from_url(url2)

        if image1 is None or image2 is None:
            return "Error: Unable to load one or both images"

        # Detect faces with rotation contingency
        image1, faces1 = detect_faces_with_rotation(image1)
        image2, faces2 = detect_faces_with_rotation(image2)

        if len(faces1) == 0 or len(faces2) == 0:
            return "Error: No faces detected in one or both images"

        # Extract face embeddings
        embeddings1 = get_face_embeddings(image1, faces1)
        embeddings2 = get_face_embeddings(image2, faces2)

        if len(embeddings1) == 0 or len(embeddings2) == 0:
            return "Error: Unable to extract face embeddings from one or both images"

        # Compare faces and calculate similarities
        for emb1 in embeddings1:
            for emb2 in embeddings2:
                similarity = compare_faces(emb1, emb2)
                if similarity * 100 > 55:
                    return "Faces match"
        return "Faces do not match"
    except Exception as e:
        print(f"Error during comparison: {e}")
        return "Error: An unexpected error occurred during processing"

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/compare", response_class=JSONResponse)
async def compare(request: Request, url1: str = Form(...), url2: str = Form(...)):
    result = compare_images_from_urls(url1, url2)
    return JSONResponse(content={"result": result})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
