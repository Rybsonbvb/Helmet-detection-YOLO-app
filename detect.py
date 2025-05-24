from ultralytics import YOLO
import easyocr
import cv2
import numpy as np

MODEL_PATH = "models/best.pt"
model = YOLO(MODEL_PATH)

ocr_reader = easyocr.Reader(['en'])

class_names = ['with helmet','without helmet','rider','number plate']

def detect_objects(img: np.ndarray,conf_threshold: float = 0.25):
    results=model(img)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)
    
    keep= scores >= conf_threshold
    return boxes[keep], scores[keep], classes[keep]

