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
    plates_centers = []       
    plates_boxes = []           
    illegal_centers = []        
    rider_boxes = []            

    for box, cls_idx in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, box)
        xmin, xmax = sorted((x1, x2))
        ymin, ymax = sorted((y1, y2))
        cx = (xmin + xmax) // 2
        cy = (ymin + ymax) // 2

        if class_names[cls_idx] == 'number plate':
            plates_centers.append((cx, cy))
            plates_boxes.append((xmin, ymin, xmax, ymax))
        elif class_names[cls_idx] == 'without helmet':
            illegal_centers.append((cx, cy))
        elif class_names[cls_idx] == 'rider':
            rider_boxes.append((xmin, ymin, xmax, ymax))

    matched_boxes = []

    for rx1, ry1, rx2, ry2 in rider_boxes:
        for il_cx, il_cy in illegal_centers:
            if rx1 <= il_cx <= rx2 and ry1 <= il_cy <= ry2:
                for (pl_cx, pl_cy), box in zip(plates_centers, plates_boxes):
                    if rx1 <= pl_cx <= rx2 and ry1 <= pl_cy <= ry2:
                        matched_boxes.append(box)

    texts = []

    for (x1, y1, x2, y2) in matched_boxes:
        cropped = img[y1:y2, x1:x2]
        results = ocr_reader.readtext(cropped)
        if results:
            best = max(results, key=lambda r: r[2])
            texts.append(best[1])
        else:
            texts.append("Nie rozpoznano tablicy")

    return texts
             
             