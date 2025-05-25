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

def extract_license_plates(img: np.ndarray, boxes: np.ndarray, classes: np.ndarray):
    plates = []
    plates_ind = []
    illegal = []
    riders = []

    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, box)
        xmin, xmax = sorted((x1, x2))
        ymin, ymax = sorted((y1, y2))
        cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2
        
        if class_names[cls] == 'number plate':
            plates.append((cx, cy))
            plates_ind.append(box)
        elif class_names[cls] == 'without helmet':
            illegal.append((cx, cy))
        elif class_names[cls] == 'rider':
            riders.append(box)

    matched_plates = []

    for rider_box in riders:
        rx1, ry1, rx2, ry2 = map(int, rider_box)
        for il_cx, il_cy in illegal:
            if rx1 <= il_cx <= rx2 and ry1 <= il_cy <= ry2:
                for i, (pl_cx, pl_cy) in enumerate(plates):
                    if rx1 <= pl_cx <= rx2 and ry2 <= pl_cy <= ry1:
                        matched_plates.append(plates_ind[i])

    plate_texts = []
    for box in matched_plates:
        x1, y1, x2, y2 = map(int, box)
        cropped = img[y1:y2, x1:x2] 
        result = ocr_reader.readtext(cropped)
    
        if result:
            best = max(result, key=lambda r: r[2])
            plate_texts.append(best[1])
        else:
            plate_texts.append("Nie rozpoznano tablicy")
    
    return plate_texts
             
             