import streamlit as st
import numpy as np
import cv2
from detect import detect_objects, extract_license_plates, class_names

st.title(" Motorcycle Helmet & Plate Detector")

img_file = st.camera_input("Zrób zdjęcie") or st.file_uploader("Lub wrzuć plik JPG/PNG", type=["jpg","png"])
if img_file:
    data = img_file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    boxes, scores, classes = detect_objects(img, conf_threshold=0.25)
    no_helmet = any(class_names[c]=='without helmet' for c in classes)

    for box, score, cls in zip(boxes, scores, classes):
        x1,y1,x2,y2 = map(int, box)
        label = f"{class_names[cls]} {score:.2f}"
        if class_names[cls]=='without helmet': 
            color = (0,0,255)
        elif class_names[cls]=='with helmet':
            color = (0,255,0)
        elif class_names[cls]=='rider':
            color = (0,165,255)
        else:
            color = (255,255,0)
             
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        cv2.putText(img, label, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if no_helmet:
        plates = extract_license_plates(img, boxes, classes)
        st.error("🚨 Wykryto osobę BEZ kasku!")
        if plates:
            st.warning(f"Znalezione tablice: {plates}")
        else:
            st.warning("Nie znaleziono tablic rejestracyjnych!")
    else:
        st.success("✅ Wszyscy mają kaski!")

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB")